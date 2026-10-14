// Boykov–Kolmogorov max-flow against two references: the brute-force minimum cut (every labelling
// of the nodes) on small graphs, and Edmonds–Karp (shortest augmenting paths) on medium ones.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "core/stereo/maxflow.hpp"

namespace {

using vl::stereo::MaxFlow;
using Cap = std::int64_t;

struct Edge {
    int i, j;
    Cap cap, rev;
};
struct RandomGraph {
    int n = 0;
    std::vector<Cap> toSource, toSink;  // tweights: paid when the node is on the sink / source side
    std::vector<Edge> edges;
};

RandomGraph randomGraph(std::mt19937& rng, int n, int m, int maxCap) {
    std::uniform_int_distribution<int> node(0, n - 1), cap(0, maxCap), coin(0, 2);
    RandomGraph g;
    g.n = n;
    for (int i = 0; i < n; ++i) {
        // Some nodes without terminal links, so that free nodes and long paths appear.
        g.toSource.push_back(coin(rng) == 0 ? 0 : cap(rng));
        g.toSink.push_back(coin(rng) == 0 ? 0 : cap(rng));
    }
    for (int k = 0; k < m; ++k) {
        const int i = node(rng), j = node(rng);
        if (i != j) g.edges.push_back({i, j, cap(rng), coin(rng) == 0 ? 0 : cap(rng)});
    }
    return g;
}

template <typename C>
void build(const RandomGraph& g, MaxFlow<C>& mf) {
    for (int i = 0; i < g.n; ++i) mf.addNode();
    for (int i = 0; i < g.n; ++i) mf.addTweights(i, C(g.toSource[size_t(i)]), C(g.toSink[size_t(i)]));
    for (const auto& e : g.edges) mf.addEdge(e.i, e.j, C(e.cap), C(e.rev));
}

// Cost of the cut where node i is on the sink side iff bit i of sinkSet is set.
Cap cutCost(const RandomGraph& g, unsigned sinkSet) {
    Cap c = 0;
    for (int i = 0; i < g.n; ++i) c += (sinkSet >> i & 1u) ? g.toSource[size_t(i)] : g.toSink[size_t(i)];
    for (const auto& e : g.edges) {
        const bool si = sinkSet >> e.i & 1u, sj = sinkSet >> e.j & 1u;
        if (!si && sj) c += e.cap;
        if (si && !sj) c += e.rev;
    }
    return c;
}

Cap edmondsKarp(const RandomGraph& g) {
    const int n = g.n + 2, s = g.n, t = g.n + 1;
    std::vector<std::vector<Cap>> c(size_t(n), std::vector<Cap>(size_t(n), 0));
    for (int i = 0; i < g.n; ++i) {
        c[size_t(s)][size_t(i)] += g.toSource[size_t(i)];
        c[size_t(i)][size_t(t)] += g.toSink[size_t(i)];
    }
    for (const auto& e : g.edges) {
        c[size_t(e.i)][size_t(e.j)] += e.cap;
        c[size_t(e.j)][size_t(e.i)] += e.rev;
    }
    Cap flow = 0;
    for (;;) {
        std::vector<int> prev(size_t(n), -1);
        prev[size_t(s)] = s;
        std::queue<int> q;
        q.push(s);
        while (!q.empty() && prev[size_t(t)] < 0) {
            const int u = q.front();
            q.pop();
            for (int v = 0; v < n; ++v)
                if (prev[size_t(v)] < 0 && c[size_t(u)][size_t(v)] > 0) {
                    prev[size_t(v)] = u;
                    q.push(v);
                }
        }
        if (prev[size_t(t)] < 0) return flow;
        Cap b = std::numeric_limits<Cap>::max();
        for (int v = t; v != s; v = prev[size_t(v)]) b = std::min(b, c[size_t(prev[size_t(v)])][size_t(v)]);
        for (int v = t; v != s; v = prev[size_t(v)]) {
            c[size_t(prev[size_t(v)])][size_t(v)] -= b;
            c[size_t(v)][size_t(prev[size_t(v)])] += b;
        }
        flow += b;
    }
}

TEST(MaxFlow, EqualsBruteForceMinCutOnSmallRandomGraphs) {
    std::mt19937 rng(1);
    for (int trial = 0; trial < 400; ++trial) {
        const int n = 2 + trial % 9;  // up to 10 nodes: 1024 cuts
        const auto g = randomGraph(rng, n, n * 2 + trial % 7, 1 + trial % 20);
        MaxFlow<Cap> mf;
        build(g, mf);
        const Cap flow = mf.maxflow();

        Cap best = std::numeric_limits<Cap>::max();
        for (unsigned s = 0; s < (1u << n); ++s) best = std::min(best, cutCost(g, s));
        ASSERT_EQ(flow, best) << "trial " << trial;

        // The cut it reports is a minimum cut.
        unsigned sinkSet = 0;
        for (int i = 0; i < n; ++i)
            if (mf.segment(i) == MaxFlow<Cap>::Sink) sinkSet |= 1u << i;
        ASSERT_EQ(cutCost(g, sinkSet), best) << "trial " << trial;
    }
}

TEST(MaxFlow, EqualsEdmondsKarpOnMediumGraphs) {
    std::mt19937 rng(7);
    for (int trial = 0; trial < 20; ++trial) {
        const int n = 150 + 10 * trial;
        const auto g = randomGraph(rng, n, 4 * n, 100);
        MaxFlow<Cap> mf;
        build(g, mf);
        const Cap flow = mf.maxflow();
        ASSERT_EQ(flow, edmondsKarp(g)) << "trial " << trial;
        // The reported partition is a cut of exactly that capacity.
        Cap c = 0;
        auto side = [&](int i) { return mf.segment(i) == MaxFlow<Cap>::Sink; };
        for (int i = 0; i < n; ++i) c += side(i) ? g.toSource[size_t(i)] : g.toSink[size_t(i)];
        for (const auto& e : g.edges) c += (!side(e.i) && side(e.j)) ? e.cap : (side(e.i) && !side(e.j)) ? e.rev : 0;
        ASSERT_EQ(c, flow) << "trial " << trial;
    }
}

TEST(MaxFlow, GridWithRealCapacitiesMatchesIntegerRun) {
    // A 4-connected grid like the ones built by graph-cut stereo, solved with double and int64
    // capacities: same flow, same cut.
    std::mt19937 rng(3);
    const int W = 30, H = 20;
    std::uniform_int_distribution<int> cap(0, 50), unary(-40, 40);
    RandomGraph g;
    g.n = W * H;
    for (int i = 0; i < g.n; ++i) {
        const int u = unary(rng);
        g.toSource.push_back(u > 0 ? u : 0);
        g.toSink.push_back(u < 0 ? -u : 0);
    }
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            if (x + 1 < W) g.edges.push_back({i, i + 1, cap(rng), cap(rng)});
            if (y + 1 < H) g.edges.push_back({i, i + W, cap(rng), cap(rng)});
        }
    MaxFlow<Cap> a;
    MaxFlow<double> b;
    build(g, a);
    build(g, b);
    const Cap fa = a.maxflow();
    EXPECT_DOUBLE_EQ(double(fa), b.maxflow());
    EXPECT_EQ(fa, edmondsKarp(g));
    for (int i = 0; i < g.n; ++i) EXPECT_EQ(int(a.segment(i)), int(b.segment(i)));
}

TEST(MaxFlow, ClearReusesTheObject) {
    std::mt19937 rng(11);
    MaxFlow<Cap> mf;
    for (int trial = 0; trial < 30; ++trial) {
        const auto g = randomGraph(rng, 8, 20, 9);
        mf.clear();
        build(g, mf);
        Cap best = std::numeric_limits<Cap>::max();
        for (unsigned s = 0; s < (1u << g.n); ++s) best = std::min(best, cutCost(g, s));
        ASSERT_EQ(mf.maxflow(), best);
    }
}

}  // namespace
