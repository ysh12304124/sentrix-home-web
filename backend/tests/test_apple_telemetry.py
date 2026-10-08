import unittest
from unittest.mock import patch

from backend.apple_telemetry import (
    _listening_ports, parse_agx_performance, parse_macmon_sample, parse_vm_stat,
)


VM_STAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               10.
Pages active:                             100.
Pages inactive:                           50.
Pages speculative:                        5.
Pages wired down:                         40.
Pages occupied by compressor:             10.
"""

AGX = '"PerformanceStatistics" = {"Alloc system memory"=1048576,"Device Utilization %"=12,"In use system memory"=524288}'


class AppleTelemetryParseTests(unittest.TestCase):
    def test_listening_ports_uses_lsof_name_fields(self):
        output = """\
p44429
f13
n*:8091
p21110
f8
n127.0.0.1:8101
p47236
f9
n*:8100
p12051
f10
n*:8771
p999
f3
n*:9999
"""
        with patch("backend.apple_telemetry._run_text", return_value=output) as run:
            found = _listening_ports({"8091", "8101", "8100", "8771"})
        self.assertEqual(found, {44429: {"8091"}, 21110: {"8101"},
                                 47236: {"8100"}, 12051: {"8771"}})
        self.assertIn("-Fpn", run.call_args.args[0])

    def test_vm_stat_used_memory_is_active_wired_and_compressor(self):
        data = parse_vm_stat(VM_STAT, page_size=16384, total_bytes=200 * 16384)
        self.assertEqual(data["system_memory_scope"], "apple_app_wired_compressed")
        self.assertEqual(data["system_memory_used_mib"], round(150 * 16384 / 1048576, 2))
        self.assertGreater(data["system_memory_total_mib"], data["system_memory_used_mib"])

    def test_vm_stat_missing_counter_is_not_zero(self):
        self.assertIsNone(parse_vm_stat("Pages free: 1.\n", page_size=16384))

    def test_agx_utilization_and_allocation(self):
        data = parse_agx_performance(AGX)
        self.assertEqual(data["gpu_utilization_pct"], 12)
        self.assertEqual(data["gpu_allocated_unified_memory_mib"], 1)
        self.assertEqual(data["gpu_in_use_unified_memory_mib"], 0.5)

    def test_agx_without_counters_stays_empty(self):
        self.assertEqual(parse_agx_performance("no counters"), {})

    def test_macmon_maps_temperature_power_and_gpu_ratio(self):
        data = parse_macmon_sample({
            "sys_power": 4.2,
            "cpu_power": 1.5,
            "gpu_power": 0.8,
            "gpu_active_ratio": 0.125,
            "temp": {"cpu_temp_avg": 41.2, "gpu_temp_avg": 38.5},
        })
        self.assertEqual(data["cpu_temperature_c"], 41.2)
        self.assertEqual(data["temperature_c"], 38.5)
        self.assertEqual(data["power_draw_w"], 4.2)
        self.assertEqual(data["power_scope"], "macmon_sys_power")
        self.assertEqual(data["gpu_utilization_pct"], 12.5)

    def test_macmon_ignores_non_objects(self):
        self.assertEqual(parse_macmon_sample([]), {})


if __name__ == "__main__":
    unittest.main()
