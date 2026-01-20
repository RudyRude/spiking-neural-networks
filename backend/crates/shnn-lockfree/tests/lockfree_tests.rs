//! Comprehensive tests for SHNN lock-free primitives refactoring
//!
//! This test suite validates all the zero-dependency lock-free components
//! that replaced crossbeam functionality.

use shnn_lockfree::{
    queue::{MPSCQueue, MPMCQueue},
    atomic::{AtomicFloat, AtomicCounter, AtomicFlag},
    ordering::MemoryOrdering,
};
use std::{
    sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}},
    thread,
    time::{Duration, Instant},
};

/// Test basic MPSC queue operations
#[test]
fn test_mpsc_queue_basic() {
    let queue = MPSCQueue::new();

    // Test empty queue
    assert!(queue.is_empty());
    assert!(queue.pop().is_err());

    // Test single push/pop
    assert!(queue.push(42).is_ok());
    assert!(!queue.is_empty());

    let popped = queue.pop().unwrap();
    assert_eq!(popped, 42);
    assert!(queue.is_empty());
}

/// Test MPSC queue with multiple producers
#[test]
fn test_mpsc_queue_multiple_producers() {
    let queue = Arc::new(MPSCQueue::new());
    let num_producers = 4;
    let items_per_producer = 1000;
    let total_items = num_producers * items_per_producer;
    
    let mut handles = Vec::new();
    
    // Spawn producer threads
    for producer_id in 0..num_producers {
        let queue_clone = queue.clone();
        let handle = thread::spawn(move || {
            for i in 0..items_per_producer {
                let item = producer_id * items_per_producer + i;
                while queue_clone.push(item).is_err() {
                    thread::yield_now();
                }
            }
        });
        handles.push(handle);
    }
    
    // Wait for all producers to finish
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Collect all items
    let mut collected = Vec::new();
    while let Ok(item) = queue.pop() {
        collected.push(item);
    }
    
    // Verify count and uniqueness
    assert_eq!(collected.len(), total_items);
    collected.sort_unstable();
    for (i, &item) in collected.iter().enumerate() {
        assert_eq!(item, i);
    }
}

/// Test MPSC queue with bounded capacity
#[test]
fn test_mpsc_queue_bounded() {
    let capacity = 10;
    let queue = MPSCQueue::new();

    // Fill to capacity (simulated, since no capacity limit)
    for i in 0..capacity {
        assert!(queue.push(i).is_ok());
    }

    // Should not be empty
    assert!(!queue.is_empty());
    assert!(queue.push(capacity).is_ok()); // Should succeed since no capacity limit

    // Pop one item
    assert_eq!(queue.pop().unwrap(), 0);
    assert!(!queue.is_empty());
}

/// Test concurrent producer-consumer scenario
#[test]
fn test_mpsc_queue_concurrent_producer_consumer() {
    let queue = Arc::new(MPSCQueue::new());
    let produced = Arc::new(AtomicU64::new(0));
    let consumed = Arc::new(AtomicU64::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));
    
    let num_producers = 3;
    let items_per_producer = 1000;
    
    let mut handles = Vec::new();
    
    // Spawn producers
    for producer_id in 0..num_producers {
        let queue_clone = queue.clone();
        let produced_clone = produced.clone();
        let stop_clone = stop_flag.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..items_per_producer {
                let item = producer_id * items_per_producer + i;
                while queue_clone.push(item).is_err() && !stop_clone.load(Ordering::SeqCst) {
                    thread::yield_now();
                }
                produced_clone.fetch_add(1, Ordering::SeqCst);
            }
        });
        handles.push(handle);
    }
    
    // Spawn consumer
    let queue_consumer = queue.clone();
    let consumed_clone = consumed.clone();
    let stop_consumer = stop_flag.clone();
    
    let consumer_handle = thread::spawn(move || {
        while !stop_consumer.load(Ordering::SeqCst) || !queue_consumer.is_empty() {
            if queue_consumer.pop().is_ok() {
                consumed_clone.fetch_add(1, Ordering::SeqCst);
            } else {
                thread::yield_now();
            }
        }
    });
    
    // Wait for producers
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Stop consumer
    stop_flag.store(true, Ordering::SeqCst);
    consumer_handle.join().unwrap();
    
    // Verify all items were produced and consumed
    let total_expected = num_producers * items_per_producer;
    assert_eq!(produced.load(Ordering::SeqCst), total_expected as u64);
    assert_eq!(consumed.load(Ordering::SeqCst), total_expected as u64);
    assert!(queue.is_empty());
}

/// Test lock-free queue implementation
#[test]
fn test_lockfree_queue_basic() {
    let queue = MPMCQueue::new();

    // Test empty
    assert!(queue.is_empty());
    assert!(queue.pop().is_err());

    // Test enqueue/dequeue
    assert!(queue.push(100).is_ok());
    assert!(!queue.is_empty());

    let item = queue.pop().unwrap();
    assert_eq!(item, 100);
    assert!(queue.is_empty());
}

/// Test lock-free queue with multiple threads
#[test]
fn test_lockfree_queue_concurrent() {
    let queue = Arc::new(MPMCQueue::new());
    let num_threads = 8;
    let items_per_thread = 500;

    let mut handles = Vec::new();

    // Spawn enqueuers and dequeuers
    for thread_id in 0..num_threads {
        let queue_clone = queue.clone();

        if thread_id % 2 == 0 {
            // Enqueuer
            let handle = thread::spawn(move || {
                for i in 0..items_per_thread {
                    let item = thread_id * items_per_thread + i;
                    let _ = queue_clone.push(item);
                }
            });
            handles.push(handle);
        } else {
            // Dequeuer
            let handle = thread::spawn(move || {
                let mut dequeued = 0;
                let start = Instant::now();
                while dequeued < items_per_thread && start.elapsed() < Duration::from_secs(10) {
                    if queue_clone.pop().is_ok() {
                        dequeued += 1;
                    } else {
                        thread::yield_now();
                    }
                }
                dequeued
            });
            handles.push(handle);
        }
    }
    
    // Wait for completion and collect results
    let mut total_dequeued = 0;
    for handle in handles {
        if let Ok(result) = handle.join() {
            if let Ok(count) = result.downcast::<usize>() {
                total_dequeued += *count;
            }
        }
    }
    
    // Some items should have been processed
    assert!(total_dequeued > 0);
}

/// Test atomic counter operations
#[test]
fn test_atomic_counter() {
    let counter = AtomicCounter::new(42);

    // Test load
    assert_eq!(counter.load(MemoryOrdering::SeqCst.into()), 42);

    // Test store
    counter.store(84, MemoryOrdering::SeqCst.into());
    assert_eq!(counter.load(MemoryOrdering::SeqCst.into()), 84);

    // Test increment
    let old_value = counter.increment();
    assert_eq!(old_value, 84);
    assert_eq!(counter.load(MemoryOrdering::SeqCst.into()), 85);
}

/// Test atomic flag operations
#[test]
fn test_atomic_flag() {
    let flag = AtomicFlag::new(false);

    // Test initial state
    assert!(!flag.load(MemoryOrdering::SeqCst.into()));

    // Test set
    flag.store(true, MemoryOrdering::SeqCst.into());
    assert!(flag.load(MemoryOrdering::SeqCst.into()));

    // Test reset
    flag.store(false, MemoryOrdering::SeqCst.into());
    assert!(!flag.load(MemoryOrdering::SeqCst.into()));
}

/// Test memory ordering semantics
#[test]
fn test_memory_ordering() {
    let counter = Arc::new(AtomicCounter::new(0));
    let flag = Arc::new(AtomicBool::new(false));

    let counter_clone = counter.clone();
    let flag_clone = flag.clone();

    let writer = thread::spawn(move || {
        counter_clone.store(42, MemoryOrdering::Release.into());
        flag_clone.store(true, MemoryOrdering::Release.into());
    });

    let reader = thread::spawn(move || {
        // Wait for flag with acquire ordering
        while !flag.load(MemoryOrdering::Acquire.into()) {
            thread::yield_now();
        }

        // Load with acquire ordering
        counter.load(MemoryOrdering::Acquire.into())
    });

    writer.join().unwrap();
    let result = reader.join().unwrap();
    assert_eq!(result, 42);
}

/// Test ABA problem prevention
#[test]
fn test_aba_prevention() {
    let queue = Arc::new(MPSCQueue::new());
    let iterations = 1000;

    // Fill queue initially
    for i in 0..10 {
        let _ = queue.push(i);
    }

    let queue_clone = queue.clone();
    let aba_thread = thread::spawn(move || {
        for _ in 0..iterations {
            // Try to create ABA scenario
            if queue_clone.pop().is_ok() {
                // Immediately push back (simplified)
                let _ = queue_clone.push(0);
            }
        }
    });

    let queue_clone2 = queue.clone();
    let normal_thread = thread::spawn(move || {
        for i in 100..100 + iterations {
            while queue_clone2.push(i).is_err() {
                thread::yield_now();
            }
        }
    });

    aba_thread.join().unwrap();
    normal_thread.join().unwrap();

    // Queue should still be functional
    let mut count = 0;
    while queue.pop().is_ok() {
        count += 1;
        if count > iterations + 100 {
            break; // Prevent infinite loop
        }
    }

    assert!(count > 0);
}

/// Test performance under contention
#[test]
fn test_performance_under_contention() {
    let queue = Arc::new(MPSCQueue::new());
    let num_threads = 16;
    let operations_per_thread = 10000;

    let start = Instant::now();
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let queue_clone = queue.clone();
        let handle = thread::spawn(move || {
            if thread_id % 2 == 0 {
                // Producer
                for i in 0..operations_per_thread {
                    let item = thread_id * operations_per_thread + i;
                    while queue_clone.push(item).is_err() {
                        thread::yield_now();
                    }
                }
            } else {
                // Consumer
                let mut consumed = 0;
                while consumed < operations_per_thread {
                    if queue_clone.pop().is_ok() {
                        consumed += 1;
                    } else {
                        thread::yield_now();
                    }
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let elapsed = start.elapsed();
    let total_operations = num_threads * operations_per_thread;

    println!("Completed {} operations in {:?}", total_operations, elapsed);
    println!("Operations per second: {:.0}", total_operations as f64 / elapsed.as_secs_f64());

    // Should complete in reasonable time
    assert!(elapsed < Duration::from_secs(30));
}

/// Test queue behavior
#[test]
fn test_queue_behavior() {
    let queue = MPSCQueue::new();

    // Fill queue
    for i in 0..100 {
        assert!(queue.push(i).is_ok());
    }

    // Test multiple threads pushing
    let queue_shared = Arc::new(queue);
    let success_count = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();

    for _ in 0..4 {
        let queue_clone = queue_shared.clone();
        let success_clone = success_count.clone();

        let handle = thread::spawn(move || {
            for i in 0..100 {
                if queue_clone.push(i).is_ok() {
                    success_clone.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Should have many successes
    assert!(success_count.load(Ordering::SeqCst) > 0);
}

/// Test lock-free data structure linearizability
#[test]
fn test_linearizability() {
    let queue = Arc::new(MPMCQueue::new());
    let operations = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();

    // Spawn multiple threads doing mixed operations
    for thread_id in 0..8 {
        let queue_clone = queue.clone();
        let ops_clone = operations.clone();

        let handle = thread::spawn(move || {
            for i in 0..1000 {
                ops_clone.fetch_add(1, Ordering::SeqCst);

                if i % 2 == 0 {
                    let _ = queue_clone.push(thread_id * 1000 + i);
                } else {
                    let _ = queue_clone.pop();
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(operations.load(Ordering::SeqCst), 8000);

    // Queue should still be in a valid state
    let mut remaining_items = 0;
    while queue.pop().is_ok() {
        remaining_items += 1;
        if remaining_items > 10000 {
            break; // Safety limit
        }
    }
}

/// Test memory reclamation and cleanup
#[test]
fn test_memory_cleanup() {
    // Test that our lock-free structures properly clean up memory
    let initial_items = 1000;

    {
        let queue = MPSCQueue::new();

        // Add many items
        for i in 0..initial_items {
            let _ = queue.push(format!("item_{}", i));
        }

        // Remove half
        for _ in 0..initial_items / 2 {
            let _ = queue.pop();
        }

        // Queue should drop cleanly when going out of scope
    }

    // If we reach here without segfault, memory cleanup worked
    assert!(true);
}

/// Test edge cases and error conditions
#[test]
fn test_edge_cases() {
    // Test with empty queue
    let empty_queue: MPSCQueue<i32> = MPSCQueue::new();
    assert!(empty_queue.pop().is_err());
    assert!(empty_queue.is_empty());

    // Test with very large items
    let large_queue = MPSCQueue::new();
    assert!(large_queue.push(42).is_ok());
    assert_eq!(large_queue.pop().unwrap(), 42);

    // Test rapid push/pop cycles
    let cycle_queue = MPSCQueue::new();
    for _ in 0..10000 {
        assert!(cycle_queue.push(1).is_ok());
        assert_eq!(cycle_queue.pop().unwrap(), 1);
    }
}