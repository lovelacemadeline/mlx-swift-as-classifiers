// Process memory as jetsam sees it: `phys_footprint` (what the per-app limit is
// charged against), its lifetime peak, and `os_proc_available_memory()` (headroom
// left before the limit). MLX's own allocator counters ride along.

import Darwin
import Foundation
import MLX
import os

struct MemorySample: Sendable, CustomStringConvertible {
    var footprint: UInt64
    var footprintPeak: UInt64
    var available: UInt64
    var mlxActive: Int
    var mlxCache: Int
    var mlxPeak: Int

    static func now() -> MemorySample {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        let ok = kr == KERN_SUCCESS
        return MemorySample(
            footprint: ok ? info.phys_footprint : 0,
            footprintPeak: ok ? UInt64(info.ledger_phys_footprint_peak) : 0,
            available: UInt64(os_proc_available_memory()),
            mlxActive: Memory.activeMemory, mlxCache: Memory.cacheMemory, mlxPeak: Memory.peakMemory)
    }

    /// Footprint plus headroom: the limit the process is running under.
    var limit: UInt64 { footprint + available }

    var description: String {
        "footprint=\(gib(footprint)) peak=\(gib(footprintPeak)) available=\(gib(available)) "
            + "limit~\(gib(limit)) mlx_active=\(gib(UInt64(mlxActive))) mlx_cache=\(gib(UInt64(mlxCache))) "
            + "mlx_peak=\(gib(UInt64(mlxPeak)))"
    }
}

func gib(_ b: UInt64) -> String { String(format: "%.2fGiB", Double(b) / 1_073_741_824) }
