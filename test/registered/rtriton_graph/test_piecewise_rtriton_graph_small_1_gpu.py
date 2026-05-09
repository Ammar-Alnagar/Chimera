import unittest

import torch

from sglang import Engine
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_rtriton_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
    run_bench_one_batch,
)

# CI Registration - Small 1-GPU tests (24GB GPU sufficient)
register_rtriton_ci(est_time=460, suite="stage-b-test-small-1-gpu")


class TestPiecewiseRtritonGraphCorrectness(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-piecewise-rtriton-graph"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)


class TestPiecewiseRtritonGraphBenchmark(CustomTestCase):

    def test_latency(self):
        prefill_latency, _, _ = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST,
            other_args=["--enable-piecewise-rtriton-graph"],
        )
        self.assertLess(prefill_latency, 0.015)


@unittest.skipIf(get_device_sm() < 100, "Test requires RTRITON SM 100 or higher")
class TestPiecewiseRtritonGraphLlama31FP4(CustomTestCase):
    """MGSM test: piecewise RTRITON graph with NVFP4 Llama3.1 8B on Blackwell."""

    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/Llama-3.1-8B-Instruct-FP4"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-rtriton-graph",
                "--quantization",
                "modelopt_fp4",
                "--mem-fraction-static",
                "0.8",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_accuracy(self):
        num_examples = 1319
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=num_examples,
            num_threads=min(num_examples, 1024),
        )
        metrics = run_eval(args)
        print(f"MGSM Accuracy: {metrics['score']:.3f}")
        self.assertGreaterEqual(metrics["score"], 0.78)


class TestPiecewiseRtritonGraphDeepSeek(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-rtriton-graph",
                "--piecewise-rtriton-graph-compiler",
                "eager",
                "--piecewise-rtriton-graph-max-tokens",
                "4096",  # should less than max_context_len
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.62)


class TestPiecewiseRtritonGraphFP8(CustomTestCase):
    """Test piecewise RTRITON graph with FP8 quantized model"""

    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/Llama-3.1-8B-Instruct-FP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-rtriton-graph",
                "--quantization",
                "modelopt_fp8",
                "--kv-cache-dtype",
                "bfloat16",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_accuracy(self):
        """Test MGSM accuracy with FP8 model"""
        num_examples = 1319
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=num_examples,
            num_threads=min(num_examples, 1024),
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.85)
        print(f"MGSM Accuracy: {metrics['score']:.3f}")


class TestPiecewiseRtritonGraphQwen25VL(CustomTestCase):
    """Test piecewise RTRITON graph with Qwen2.5-VL-7B-Instruct model"""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-rtriton-graph",
                "--piecewise-rtriton-graph-compiler",
                "eager",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        """Test GSM8K accuracy with 8-shot setting"""
        num_examples = 2000

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=num_examples,
            num_threads=min(num_examples, 1024),
        )

        metrics = run_eval(args)
        print(f"GSM8K Accuracy: {metrics['score']:.3f}")

        self.assertGreaterEqual(metrics["score"], 0.70)


class TestPiecewiseRtritonGraphInternVL25(CustomTestCase):
    """Test piecewise RTRITON graph with InternVL2.5-8B-Instruct model"""

    @classmethod
    def setUpClass(cls):
        cls.model = "OpenGVLab/InternVL2_5-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-rtriton-graph",
                "--piecewise-rtriton-graph-compiler",
                "eager",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        """Test GSM8K accuracy with 8-shot setting"""
        num_examples = 2000

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=num_examples,
            num_threads=min(num_examples, 1024),
        )

        metrics = run_eval(args)
        print(f"GSM8K Accuracy: {metrics['score']:.3f}")

        self.assertGreaterEqual(metrics["score"], 0.70)


class TestPiecewiseRtritonGraphQwen25VLEmbedding(CustomTestCase):
    """Test piecewise RTRITON graph with Qwen2.5-VL-3B-Instruct embedding model"""

    def test_embedding(self):
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        chat_template = get_chat_template_by_model_path(model_path)
        text = f"{chat_template.image_token}What is in this picture? Answer: "

        engine = Engine(
            model_path=model_path,
            enable_multimodal=True,
            is_embedding=True,
            enable_piecewise_rtriton_graph=True,
            piecewise_rtriton_graph_compiler="eager",
        )
        out = engine.encode([text], image_data=[DEFAULT_IMAGE_URL])[0]["embedding"]
        engine.shutdown()
        self.assertGreater(len(out), 0)

        engine = Engine(
            model_path=model_path,
            enable_multimodal=True,
            is_embedding=True,
            enable_piecewise_rtriton_graph=False,
        )
        out_without_pcg = engine.encode([text], image_data=[DEFAULT_IMAGE_URL])[0][
            "embedding"
        ]
        engine.shutdown()
        self.assertGreater(len(out_without_pcg), 0)

        self.assertTrue(
            torch.allclose(torch.tensor(out), torch.tensor(out_without_pcg))
        )


if __name__ == "__main__":
    unittest.main()
