import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import plan  # noqa: E402
from plan import GPUS, classify, command, native_budget  # noqa: E402


def tier(gpu, n, host=0):
    return classify(GPUS[gpu], n, host)[0]


class Args:
    def __init__(self, gpu, n=1, host_mem=0, mem_bw=0, mode='latency'):
        self.gpu, self.n, self.host_mem, self.mem_bw, self.mode = gpu, n, host_mem, mem_bw, mode


class TestNumbers(unittest.TestCase):
    def test_checkpoint_split(self):
        self.assertEqual(plan.BACKBONE_GB + plan.ENGRAM_GB, 510)

    def test_routed_read_per_token(self):
        self.assertAlmostEqual(plan.ROUTED_READ_GB, 4.5, delta=0.1)

    def test_h20_8card_budget(self):
        on = native_budget(GPUS['h20-96'], 8, True, 0.88)
        off = native_budget(GPUS['h20-96'], 8, False, 0.88)
        self.assertAlmostEqual(on['kv'], 20.7, delta=0.1)
        self.assertAlmostEqual(off['kv'], 46.1, delta=0.1)
        self.assertGreater(off['kv_mtokens'], on['kv_mtokens'])

    def test_kv_bytes_by_arch(self):
        self.assertEqual(plan.kv_bytes(GPUS['b200']), 890)
        self.assertEqual(plan.kv_bytes(GPUS['h20-96']), 1790)

    def test_decode_ceiling(self):
        self.assertAlmostEqual(plan.decode_ceiling(410), 91, delta=1)


class TestTiers(unittest.TestCase):
    def test_h20_8card_is_tier1(self):
        self.assertEqual(tier('h20-96', 8), 1)

    def test_h20_4card_needs_host_memory(self):
        self.assertEqual(tier('h20-96', 4, 384), 2)
        t, notes = classify(GPUS['h20-96'], 4, 128)
        self.assertNotEqual(t, 2)
        self.assertTrue(any('256 GB' in x for x in notes))

    def test_2x_b200_by_per_card_budget(self):
        # 显存合计 360 GB 超过骨干，但 Engram 留显存后每卡 KV 不够
        self.assertEqual(tier('b200', 2, 512), 2)

    def test_4x_pro6000_native(self):
        # 社区配方：SGLang 原生 TP4，Engram 走 NVMe 加内存缓存
        self.assertEqual(tier('rtx-pro-6000', 4, 128), 2)
        t, notes = classify(GPUS['rtx-pro-6000'], 4, 64)
        self.assertNotEqual(t, 2)
        self.assertEqual(tier('rtx-pro-6000', 2, 512), 3)

    def test_consumer_cards_go_kt(self):
        self.assertEqual(tier('rtx5070ti', 2, 512), 3)
        self.assertEqual(tier('rtx-pro-6000', 1, 499), 3)
        t, notes = classify(GPUS['rtx4090'], 1, 384)
        self.assertEqual(t, 3)
        self.assertTrue(any('Triton' in x for x in notes))

    def test_old_arch_falls_to_gguf(self):
        self.assertEqual(tier('a100-80', 8), 4)
        t, notes = classify(GPUS['rtx3090'], 1, 512)
        self.assertEqual(t, 4)
        self.assertTrue(any('SM86' in x for x in notes))

    def test_small_box_is_api(self):
        self.assertEqual(tier('rtx5090', 1, 64), 5)


class TestOutput(unittest.TestCase):
    def test_tier1_latency_uses_dspark(self):
        cmd = command(1, GPUS['h20-96'], 8, 'latency')
        self.assertIn('--tp 8', cmd)
        self.assertIn('DSPARK', cmd)
        self.assertNotIn('--attention-backend', cmd)

    def test_tier2_puts_engram_on_cpu(self):
        cmd = command(2, GPUS['h20-96'], 4, 'latency')
        self.assertIn('"placement": "cpu"', cmd)
        self.assertIn('--max-running-requests 16', cmd)
        self.assertIn('--cuda-graph-max-bs 16', cmd)

    def test_pro6000_native_command(self):
        cmd = command(2, GPUS['rtx-pro-6000'], 4, 'latency')
        self.assertIn('docker compose', cmd)

    def test_public_rows(self):
        out = '\n'.join(plan.report(Args('h20-141', 8)))
        self.assertIn('#46310', out)
        self.assertIn('984', out)
        out = '\n'.join(plan.report(Args('h20-96', 8)))
        self.assertIn('近邻', out)
        out = '\n'.join(plan.report(Args('rtx-pro-6000', 4, 128)))
        self.assertIn('705', out)

    def test_gguf_pick(self):
        out = '\n'.join(plan.report(Args('rtx4090', 1, 128)))
        self.assertIn('UD-IQ1_S', out)
        out = '\n'.join(plan.report(Args('a100-80', 8, 0)))
        self.assertIn('UD-Q4_K_XL', out)

    def test_kt_report_lists_known_experts(self):
        out = '\n'.join(plan.report(Args('rtx-pro-6000', 1, 499, 300)))
        self.assertIn('90 个', out)
        self.assertIn('tok/s', out)


if __name__ == '__main__':
    unittest.main()
