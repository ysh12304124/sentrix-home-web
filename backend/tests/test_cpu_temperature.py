"""CPU package temperature is read from the CPU hwmon, not NVMe or the iGPU."""
from pathlib import Path
import tempfile
import unittest

from backend.hardware import read_cpu_temperature_c


class CpuTemperatureTest(unittest.TestCase):
    def test_prefers_k10temp_tctl_over_nvme(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nvme = root / "hwmon0"
            nvme.mkdir()
            (nvme / "name").write_text("nvme\n", encoding="ascii")
            (nvme / "temp1_input").write_text("42850\n", encoding="ascii")
            cpu = root / "hwmon2"
            cpu.mkdir()
            (cpu / "name").write_text("k10temp\n", encoding="ascii")
            (cpu / "temp1_input").write_text("69125\n", encoding="ascii")
            (cpu / "temp1_label").write_text("Tctl\n", encoding="ascii")
            self.assertEqual(read_cpu_temperature_c(root), 69.125)

    def test_missing_sensor_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_cpu_temperature_c(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
