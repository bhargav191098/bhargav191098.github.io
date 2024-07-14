---
title: "Experiments with Learned Indexing"
description: "Adding Learned Indexing functionality in DuckDB!"
pubDate: "Jul 14 2024"
heroImage: "/index_cover.jpg"
---

I kinda wanted to write about this project for sometime now. Bit of a block! So just decided to sit down, think from first principles and build a flow.

### 1. Let's think about indexing

To begin with, what do we know about indexing? We have extensive data, often in the form of numerous rows. A naive approach to finding a specific row would involve sequentially examining each entry. However, decades of data structures and algorithms have taught us to pre-arrange data for efficient retrieval. One common method is hashing, where we map a value to a memory address. Although hashing offers O(1) time complexity, it's not always feasible.

Instead, we use tree structures to organize data. For instance, when you create an index for a database column, the column values become node keys in a tree, with the corresponding records stored in associated nodes. Many integer-based indexes maintain sorted order, taking on some complexity during the insertion time and offload that complexity during retrieval time. To take a step back and abstract it even more, we reduce the entropy of the data we have. You know where I'm going this, right? <i><b>Where there is data entropy, Machine Learning will find an application there.</i></b>


So what we did with B+, B Tree & Binary Search Trees is to fit the data into an ordering we know will give us good retrieval speed. For instance, the rules for BST are fairly straightforward: the left part is smaller, the right part should be bigger, there should be only two children and the rules apply recursively downwards. The concept of Learned Indexing is to basically think - can we figure out these rules tailor made for each dataset and not throw a generalistic data structure at the data.


I'm going to use an extreme example to showcase this. In fact, the original authors of the paper use this in paper explanation videos. 

![Learned Index example extreme](/LearnedIndex1.png)

In this extreme example, every integer entry from 1000 is present as a key. If we use a B-tree to index this data, we need to store all the keys in the data nodes, in addition to the pointer nodes to the next level of nodes. Basically chase the pointers to get to the data. Instead what the authors of the learned index propose is that, if we could identify that the keys that we need to index are a sequential set of numbers starting from 1000 and going on till 1000+n, we can simply store the starting point, and then compute the offset with each index key and get to the key. Neat right? But this rests upon two assumptions : the storage is sequential, pre-sorted and the pattern exhibited by the data is rather simplistic. To generalize this pattern, what we need to find is the CDF of the data distribution. Let's take a harder column from the same table - the date.

![CDF](/LearnedIndex2.png)

But the CDFs can be diverse and multi-dimensional in nature. In pure machine learning terms, we cannot throw one single model at a CDF and solve it. It might overfit to a particular trend and it might be very bad at accommadating shifts in key distribution. To counter this, the authors come up with the concept of Recursive Model Index. It is inspired from the structure of B+ tree. Each node has a linear regressor and traversing the tree basically means performing floating point multiplications. From the model perspective, what this means is that the CDF is broken piece wise and each node level model takes care of that region. 

Indexing is the art of being accurate and exact. If your next question was how does the art of approximation (machine learning) go well with this? As we know each prediction has an accuracy rate. We do a simple linear search in the final mile ensuring the accuracy. All of this is the story of the fundamental learned indexing. Post that came ALEX - an updatable learned index. It bought forth model based insertions and a ternary search in the last mile - so it is blazing fast.

### 2. DuckDB Extension for ALEX

My curiosity was in applying this in a real world database and see how it fares. I picked DuckDB because it was a beautiful fit. It was the SQLite for OLAP workloads that had an OLTP style index for OLTP workloads. It is designed to be light weight and DuckDB extensions is literally, the power of sun in the palm of your hands.


The Intelligent Duck Extension adds the functionality of learned indexing for DuckDB. DuckDB is a in-process, lightweight OLAP database <b>duckdb</b>. Termed as the SQLite for OLAP, DuckDB has made many optimizations to make the storage of large volumes of data compact. DuckDB defines its data unit as a vector that the processing engine handles. While storing, depending on the nature of the data, the type of vectors chosen are flattened, structural type, and so on. With such extensive optimizations in place, the problem of OLTP style is complicated. There is no conventional notion of row-based pointers but rather an abstraction of the vector-based idea on which their table structure is based. <br>

DuckDB added the support for <b>ART indexing</b> to provide a feature for <b>key constraints</b> and speed up point lookup queries. It <b>does <a href="https://duckdb.org/docs/guides/performance/indexing.html">not</a></b> enhance the performance of joins or sorting in any way. With such a gap in the actual utility of the indices, integrating the learned indexes should be desirable. This would also allow observing the performance of a learned index in a robust, commercial OLAP database. For the Learned Index, we pick ALEX \cite{alex}, which is an updatable learned index. One of the primary reasons for picking this was the rather dynamic nature of the index itself. It can be updated, and the SIGMOD paper has shown great promise. On read-only workloads, the ALEX indexing structure has a <b>3x higher throughput</b> and <b>8000x</b> smaller index size than ART. To summarize, the objectives of the project are as follows: <br>

- Improve the index size required for this OLTP-style index in the OLAP DuckDB database.
- Have a faster lookup time than ART-Indexing, which is commercially available.
- Observe the performance of a popular learned index in a standard system without extreme hardware advantage.

The original ALEX makes use of `x86 SIMD Vectorized` operations. To run it on mac arm64 architecture, had to swap out some vectorized calls for serial calls and it caused a dip in the bulk loading time post 1 million keys.

### 3. Experimental Evaluation

We compare <b>ALEX</b> with the <b>B+Tree</b>, the <b>native duckdb ART indexing</b> (Adaptive Radix Tree), using four benchmarking datasets mentioned in the ALEX paper: <b>Lognormal, Longlat, Longitudes \& YCSB</b> (2 Double and 2 64 Bit Int key datasets). <br>

#### 3.1 The setup
Our evaluations are conducted on a MacBook Pro equipped with an Apple M1 chip, featuring an 8-core CPU and 16GB of RAM. This decision roots from our inclination in testing the efficacy of Learned indexing in standard systems. This is a marked difference from the strong x86 hardware devices that are used in the original ALEX benchmarking. Another marked difference is that the ART index used in the original ALEX benchmarking is an open source implementation. DuckDB has optimized the ART Indexing for specific operating systems. So we obtained some interesting results. We feel these would be an great litmus test to the applicability of learned indices. <br>


#### 3.2 Time to lookup N keys - ART strikes back

Deviating from the standard throughput perspective to benchmarking, we took another approach. We record the amount of time in seconds to look up N keys. We increase the N value in steps of 10.<br>

The ALEX data structure is queried for N keys. But how does it fare for the ART index. We fallback to using the query interface for selecting a value from the table. So we run N-lookup calls. But each of these N lookup calls passes through the query planning stage and then N materialization stages. This is penalizing the ART evaluation unnecessarily. With these N materialization and N query plans in place, the ART indexing lookup are very slow compared to ALEX.

It was holistically not the right way to evaluate a great system like DuckDB. We discovered that the SELECT query with WHERE key in (..) also used ART index if one exists. So we materialized all the N keys into this query and made a single call to the DB with 1 query plan for the lookup of N-keys. With such a strategy only one query plan is made and only one materialization happens. The results showed how optimized the DuckDB ART is compared to the open source implementation. The DuckDB ART \textbf{matches} the lookup time of ALEX. But it should also be considered that ALEX is also constrained because of the absence of SIMD Vectorization operations.

#### 3.3 Memory consumption

To benchmark the memory consumption of the three types of indexes considered - we consider the memory required to hold the index data structure plus any auxillary data space that is used by the index. For getting the memory used by the ART indexing in duckdb we used the in-built query "select * from duckdb_memory()". The ALEX index is almost half the size of the ART index and B+ tree is the most expensive in terms of memory.

![Benchmark graphs](/benchmarks.png)

### 4. Wrapping up

There is a much more comprehensive way to do this. Rather than using DuckDB extension, one could have integrated indexing from duck db query plans. But that is thinking from DuckDB's perspective. For a 1.5 month long project keeping it light-weight via Extensions seemed like a better option!

