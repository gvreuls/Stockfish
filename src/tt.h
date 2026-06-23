/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef TT_H_INCLUDED
#define TT_H_INCLUDED

#include <tuple>

#include "misc.h"
#include "memory.h"
#include "types.h"

namespace Stockfish {

class ThreadPool;

// There is only one global hash table for the engine and all its threads. For chess in particular, we even allow racy
// updates between threads to and from the TT, as taking the time to synchronize access would cost thinking time and
// thus elo. As a hash table, collisions are possible and may cause chess playing issues (bizarre blunders, faulty mate
// reports, etc). Fixing these also loses elo; however such risk decreases quickly with larger TT size.
//
// We clearly separate Data, a local copy of an entry, from Writer, which writes to the global table.


// A copy of the data already in an entry (possibly collided). Probes and reads are racy and non-atomic,
// possibly resulting in inconsistent data.
class TranspositionTable {
   public:
    class Writer;

    struct Data {
        Move  move;
        Value value, eval;
        Depth depth;
        Bound bound;
        bool  is_pv;

        // clang-format off
        constexpr Data() :
            move(Move::none()),
            value(VALUE_NONE),
            eval(VALUE_NONE),
            depth(DEPTH_NONE),
            bound(BOUND_NONE),
            is_pv(false) {}
    
        constexpr Data(Move m, Value v, Value ev, Depth d, Bound b, bool pv) :
            move(m),
            value(v),
            eval(ev),
            depth(d),
            bound(b),
            is_pv(pv) {}
        // clang-format on
    };

    using ReadProbe  = std::tuple<bool, Data>;
    using WriteProbe = std::tuple<bool, Data, Writer>;

    class Cluster {
       public:
        static constexpr int Size    = 3;
        static constexpr int PadSize = 2;

       private:
        // Entry struct is the 8 bytes transposition table entry, defined as:
        //
        // depth       8 bit
        // pv node     1 bit
        // bound type  2 bit
        // generation  5 bit
        // move       16 bit
        // value      16 bit
        // evaluation 16 bit
        //
        // These fields are in the same order as accessed by Table::probe(), since memory is fastest sequentially.
        // Equally, the store order in save() matches this order.
        //
        // We use `bool(depth8)` as the cheap internal occupancy check, corresponding to `depth == DEPTH_NONE`
        // externally, so we offset the internal depth by DEPTH_NONE.
        //
        // Pv, bound and generation are packed in a single byte.
        //
        // The 16 bit key is stored in a different Cluster member for faster search access.
        struct Entry {
            static constexpr u8 GENERATION_BITS = 5;
            static constexpr u8 GENERATION_MASK = (1 << GENERATION_BITS) - 1;
            static constexpr u8 BOUND_SHIFT     = GENERATION_BITS;
            static constexpr u8 BOUND_MASK      = 0b11 << BOUND_SHIFT;
            static constexpr u8 PV_SHIFT        = BOUND_SHIFT + 2;
            static constexpr u8 PV_MASK         = 1 << PV_SHIFT;

            RelaxedAtomic<u8>   depth8;
            RelaxedAtomic<u8>   genBound8;
            RelaxedAtomic<Move> move16;
            RelaxedAtomic<i16>  value16;
            RelaxedAtomic<i16>  eval16;

            constexpr Data read() const {
                return Data{Move(move16),
                            Value(value16),
                            Value(eval16),
                            Depth(DEPTH_NONE + depth8),
                            Bound((genBound8 & BOUND_MASK) >> BOUND_SHIFT),
                            bool(genBound8 & PV_MASK)};
            }

            constexpr bool is_occupied() const { return bool(depth8); }

            constexpr u8 relative_age(u8 curr_generation) const {
                // Returns this entry's age. We count generations like clocks count hours,
                // i.e. we require 0 - 1 == 31. Unsigned subtraction guarantees the required
                // borrowing regardless of the upper pv/bound bits.
                return (curr_generation - genBound8) & GENERATION_MASK;
            }
        };

        RelaxedAtomic<u16> key16[Size];
        u8                 padding[PadSize];  //  Explicit padding.
        Entry              entry[Size];

        ReadProbe probe_read(Key key, u8 curr_generation) const {
            const auto [hit, index] = probe(key, curr_generation);

            if (hit)
                // This gap is the main place for read races.
                // After `read()` completes that copy is final, but may be self-inconsistent.
                return {entry[index].is_occupied(), entry[index].read()};

            return {false, {}};
        }
        WriteProbe probe_write(Key key, u8 curr_generation) {
            const auto [hit, index] = probe(key, curr_generation);

            if (hit)
                // This gap is the main place for read races.
                // After `read()` completes that copy is final, but may be self-inconsistent.
                return {entry[index].is_occupied(), entry[index].read(),
                        Writer{key16[index], entry[index]}};

            return {false, {}, Writer{key16[index], entry[index]}};
        }

        std::tuple<bool, int> probe(Key key, u8 curr_generation) const;

        friend Writer;
        friend TranspositionTable;
    };

    class Writer {
       public:
        Writer() = delete;

        void write(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, u8 curr_generation);

        void penalize(int penalty) {  // decrement stored depth by the penalty
            // guard against racy underflows, default to "unoccupied"
            entry->depth8 = std::max(int(entry->depth8) - penalty, 0);
        }

       private:
        constexpr Writer(RelaxedAtomic<u16>& k, Cluster::Entry& e) :
            key16{&k},
            entry{&e} {}

        RelaxedAtomic<u16>* key16;
        Cluster::Entry*     entry;

        friend TranspositionTable;
    };


    ~TranspositionTable() { aligned_large_pages_free(table); }

    void resize(usize mbSize, ThreadPool& threads);  // Set TT size in MiB
    void clear(ThreadPool& threads);                 // Re-initialize memory, multithreaded

    // This must be called at the beginning of each root search to track entry aging
    void new_search() {
        ++generation8;
        // Don't overflow into the other bits of Entry::genBound8
        generation8 &= Cluster::Entry::GENERATION_MASK;
    }

    // The current age, used when writing new data to the TT
    constexpr u8 generation() const { return generation8; }

    // Approximate what fraction of entries (permille) have been written to during this root search
    int hashfull(int maxAge = 0) const;

    // `probe_*` are primary methods: given a board position, we lookup its entry in the table, and return a tuple of:
    //   1) whether the entry already had data on this position
    //   2) a copy of the prior data, if any (may be self-inconsistent due to read races)
    //   3) a writer object to the entry for probe_write()
    ReadProbe  probe_read(Key key) const { return first_entry(key)->probe_read(key, generation8); }
    WriteProbe probe_write(Key key) { return first_entry(key)->probe_write(key, generation8); }

    // The hash function; its only external use is memory prefetching
    constexpr Cluster* first_entry(Key key) const { return &table[mul_hi64(key, clusterCount)]; }

   private:
    usize    clusterCount;
    Cluster* table       = nullptr;
    u8       generation8 = 0;
};

static_assert(sizeof(TranspositionTable::Cluster) == 32,
              "Suboptimal TranspositionTable::Cluster size");


}  // namespace Stockfish

#endif  // #ifndef TT_H_INCLUDED
