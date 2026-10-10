import importlib.util
from pathlib import Path
import tomllib
import unittest
from unittest import mock

from test_container_environment import runtime
from test_environment_policy import policy, snapshot


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("codex_configuration", ROOT / "environment/codex/configure.py")
configuration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configuration)


class CodexConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.template = (ROOT / "environment/codex/config.toml").read_text()

    def test_shared_provider_reads_the_key_from_environment(self):
        config = tomllib.loads(configuration.render_config(self.template))
        provider = config["model_providers"][config["model_provider"]]
        self.assertEqual(provider["base_url"], "https://s2api.top")
        self.assertEqual(provider["env_key"], "OPENAI_API_KEY")
        self.assertNotIn("model", config)

    def test_model_is_quoted_as_data(self):
        model = 'model"\nmodel_provider = "other'
        config = tomllib.loads(configuration.render_config(self.template, model))
        self.assertEqual(config["model"], model)
        self.assertEqual(config["model_provider"], "sdc")

    def test_personal_key_is_not_written_to_configuration(self):
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-private-value"}):
            rendered = configuration.render_config(self.template, "test-model")
        self.assertNotIn("test-private-value", rendered)

    def test_codex_updates_do_not_fail_environment_checks(self):
        baseline = {"node": "24.21.0", "npm": "11.19.0"}
        with mock.patch.object(runtime.subprocess, "check_output", side_effect=["v24.21.0\n", "11.19.0\n", "codex-cli 9.0.0\n"]):
            self.assertEqual(runtime.check_agent_tools(baseline)["codex"], "9.0.0")

    def test_node_version_remains_fixed(self):
        baseline = {"node": "24.21.0", "npm": "11.19.0"}
        with mock.patch.object(runtime.subprocess, "check_output", return_value="v26.0.0\n"):
            with self.assertRaisesRegex(ValueError, "node:"):
                runtime.check_agent_tools(baseline)

    def test_personal_env_cannot_be_committed(self):
        state = snapshot(extra={".env": "OPENAI_API_KEY=test-private-value\n"})
        with self.assertRaisesRegex(policy.PolicyError, "untracked .env"):
            policy.check(state, [".env"], "owner", "owner")


if __name__ == "__main__":
    unittest.main()
