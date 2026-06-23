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

#include "tt.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "memory.h"
#include "misc.h"
#include "syzygy/tbprobe.h"
#include "thread.h"

namespace Stockfish {


// Looks up the current position in the cluster.
// It returns true if the key is found (which may be a collision) and the cluster index of the found entry.
std::tuple<bool, int> TranspositionTable::Cluster::probe(Key key, u8 curr_generation) const {

    const u16 new_key16 = u16(key);  // Use the low 16 bits as key inside the cluster

    for (int i = 0; i < Cluster::Size; ++i)
        if (key16[i] == new_key16)
            return {true, i};

    // Find an entry to be replaced according to the replacement strategy
    int replace = 0;
    for (int i = 1; i < Cluster::Size; ++i)
        if (entry[replace].depth8 - 8 * entry[replace].relative_age(curr_generation)
            > entry[i].depth8 - 8 * entry[i].relative_age(curr_generation))
            replace = i;

    return {false, replace};
}


void TranspositionTable::Writer::write(
  Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, u8 curr_generation) {
    // Preserve the old ttmove if we don't have a new one
    if (m || u16(k) != *key16)
        entry->move16 = m;

    // Overwrite less valuable entries (cheapest checks first)
    if (b == BOUND_EXACT || u16(k) != *key16 || d - DEPTH_NONE + 2 * pv > entry->depth8 - 4
        || entry->relative_age(curr_generation))
    {
        assert(d > DEPTH_NONE);
        assert(d - DEPTH_NONE < 256);
        assert(curr_generation <= Cluster::Entry::GENERATION_MASK);  // TT::new_search() plays nice

        *key16           = u16(k);
        entry->depth8    = u8(d - DEPTH_NONE);
        entry->genBound8 = u8(curr_generation | b << Cluster::Entry::BOUND_SHIFT
                              | u8(pv) << Cluster::Entry::PV_SHIFT);
        entry->value16   = i16(v);
        entry->eval16    = i16(ev);
    }
    // Secondary aging. Important for elementary mate finding.
    // (*Scaler) Secondary aging on entries relevant to singular extensions
    // generally scales poorly and requires VVLTC verification.
    else if (entry->depth8 + DEPTH_NONE >= 5
             && Bound((entry->genBound8 & Cluster::Entry::BOUND_MASK)
                      >> Cluster::Entry::BOUND_SHIFT)
                  != BOUND_EXACT)
    {
        const i16 v16 = entry->value16;
        if (std::abs(v16) < VALUE_INFINITE && is_decisive(v16))
            entry->depth8 = std::max(int(entry->depth8) - 1,
                                     0);  // guard against racy underflows, default to "unoccupied"
    }
}


void TranspositionTable::resize(usize mbSize, ThreadPool& threads) {
    aligned_large_pages_free(table);

    clusterCount        = mbSize * 1024 * 1024 / sizeof(Cluster);
    const usize ttBytes = clusterCount * sizeof(Cluster);

    // Request 1GB pages if we'd get at least eight per NUMA node, to avoid
    // memory oversubscription
    const bool hugePageHint = ttBytes >= threads.numa_nodes() * HugePageSize * 8;

    table = static_cast<Cluster*>(aligned_large_pages_alloc_with_hint(ttBytes, hugePageHint));

    if (!table)
    {
        std::cerr << "Failed to allocate " << mbSize << "MB for transposition table." << std::endl;
        exit(EXIT_FAILURE);
    }

    clear(threads);
}


// Initializes the entire transposition table to zero,
// in a multi-threaded way.
void TranspositionTable::clear(ThreadPool& threads) {
    generation8             = 0;
    const usize threadCount = threads.num_threads();

    std::vector<usize> threadToNuma = threads.get_bound_thread_to_numa_node();

    std::vector<usize> order(threadCount);
    std::iota(order.begin(), order.end(), 0);

    // To promote good NUMA distribution (esp. with huge pages), we permute threads so that
    // all threads in a NUMA node clear a contiguous region of the TT.
    if (threadToNuma.size() == threadCount)
    {
        std::stable_sort(order.begin(), order.end(), [&threadToNuma](usize t1, usize t2) {
            return threadToNuma.at(t1) < threadToNuma.at(t2);
        });
    }

    for (usize i = 0; i < threadCount; ++i)
    {
        threads.run_on_thread(order[i], [this, i, threadCount]() {
            // Each thread will zero its part of the hash table
            const usize stride = clusterCount / threadCount;
            const usize start  = stride * i;
            const usize len    = i + 1 != threadCount ? stride : clusterCount - start;

            std::memset(static_cast<void*>(&table[start]), 0, len * sizeof(Cluster));
        });
    }

    for (usize i = 0; i < threadCount; ++i)
        threads.wait_on_thread(i);
}


// Returns an approximation of the hashtable
// occupation during a search. The hash is x permill full, as per UCI protocol.
// Only counts entries which are younger than maxAge.
int TranspositionTable::hashfull(int maxAge) const {
    int cnt = 0;
    for (int i = 0; i < 1000; ++i)
        for (int j = 0; j < Cluster::Size; ++j)
            cnt += table[i].entry[j].is_occupied()
                && table[i].entry[j].relative_age(generation8) <= maxAge;

    return cnt / Cluster::Size;
}


}  // namespace Stockfish
