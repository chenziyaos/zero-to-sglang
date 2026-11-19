"""python3 -m unittest test_deepseek.py -v"""
import os
import unittest

import align_bench as ab
import check_config as cc

HERE = os.path.dirname(os.path.abspath(__file__))


class Engram(unittest.TestCase):
    def test_default(self):
        self.assertEqual(cc.check_engram({}), cc.DEFAULT)

    def test_peer_read_needs_gpu(self):
        with self.assertRaisesRegex(ValueError, "needs placement 'gpu'"):
            cc.check_engram({'placement': 'cpu', 'lookup': 'peer_read'})
        self.assertEqual(cc.check_engram({'lookup': 'peer_read'})['placement'], 'gpu')

    def test_per_node_needs_cpu(self):
        with self.assertRaisesRegex(ValueError, "per_node"):
            cc.check_engram({'replication': 'per_node'})
        cc.check_engram({'placement': 'cpu', 'replication': 'per_node'})

    def test_collective_alias(self):
        self.assertEqual(cc.check_engram({'lookup': 'collective'})['lookup'], 'auto')

    def test_invalid_value(self):
        with self.assertRaisesRegex(ValueError, 'invalid'):
            cc.check_engram({'placement': 'disk'})


class Backend(unittest.TestCase):
    def test_mega_on_dp_attention(self):
        b, why = cc.resolve_backend('deepseek_v41', tp=4, attn_dp=4)
        self.assertEqual((b, why), ('flashmla_mega', []))

    def test_attention_tp_falls_back(self):
        b, why = cc.resolve_backend('deepseek_v41', tp=4, attn_dp=1)
        self.assertEqual(b, 'flashmla')
        self.assertIn('64 头', why[0])

    def test_attention_tp_ok_under_pd(self):
        self.assertEqual(cc.resolve_backend('deepseek_v41', tp=4, disagg='decode')[0], 'flashmla_mega')

    def test_prefill_cp_not_tp(self):
        self.assertEqual(cc.resolve_backend('deepseek_v41', tp=4, prefill_cp=True)[0], 'flashmla_mega')

    def test_v4_and_hopper(self):
        self.assertEqual(cc.resolve_backend('deepseek_v4')[0], 'flashmla')
        self.assertEqual(cc.resolve_backend('deepseek_v41', sm=90)[0], 'flashmla')
        self.assertEqual(cc.resolve_backend('deepseek_v41', kv_dtype='bf16')[0], 'flashmla')

    def test_plan_rl(self):
        out = '\n'.join(cc.plan(type('A', (), dict(
            model='deepseek_v41', device='cuda', sm=100, kv_dtype='fp8_e4m3', disagg='null',
            tp=4, attn_dp=4, attn_cp=1, prefill_cp=False, layout=None,
            engram='{"placement":"cpu","lookup":"peer_read"}', rl=True,
            deterministic=False, dspark=False))()))
        self.assertIn('启动会失败', out)
        self.assertIn('--no-enable-decoder-swa-bounded-replay', out)


class Align(unittest.TestCase):
    def test_lmsys_vs_deepinfer(self):
        rows = ab.diff(*(ab.json.load(open(os.path.join(HERE, 'data', f)))
                         for f in ('deepinfer_readme.json', 'lmsys_sglang.json')))
        names = [r[0] for r in rows]
        self.assertEqual(len(rows), 6)  # 四项不同 + 两项缺失
        self.assertIn('Engram 表放置', names)
        self.assertTrue(any('缺失' in r[3] for r in rows))

    def test_same(self):
        cfg = {k: 1 for k, _, _ in ab.ITEMS}
        self.assertEqual(ab.diff(cfg, dict(cfg)), [])


if __name__ == '__main__':
    unittest.main()
