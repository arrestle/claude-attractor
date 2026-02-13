"""Tests for ExecutionEnvironment abstraction.

Covers:
- LocalEnvironment: read, write, exec_shell, glob, file_exists, is_file
- DockerEnvironment: command building, start/stop lifecycle
- Tool integration: set_environment swaps backend transparently
- Backward compatibility: all existing tool behavior unchanged
"""

from __future__ import annotations

import os

import pytest

from attractor_agent.environment import (
    DockerEnvironment,
    ExecutionEnvironment,
    LocalEnvironment,
    ShellResult,
)
from attractor_agent.tools.core import (
    get_environment,
    set_environment,
)

# ================================================================== #
# LocalEnvironment
# ================================================================== #


class TestLocalEnvironment:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        env = LocalEnvironment()
        content = await env.read_file(str(f))
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        f = tmp_path / "out.txt"
        env = LocalEnvironment()
        await env.write_file(str(f), "written content")
        assert f.read_text() == "written content"

    @pytest.mark.asyncio
    async def test_write_creates_parents(self, tmp_path):
        f = tmp_path / "sub" / "dir" / "file.txt"
        env = LocalEnvironment()
        await env.write_file(str(f), "nested")
        assert f.read_text() == "nested"

    @pytest.mark.asyncio
    async def test_file_exists(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("yes")
        env = LocalEnvironment()
        assert await env.file_exists(str(f)) is True
        assert await env.file_exists(str(tmp_path / "nope")) is False

    @pytest.mark.asyncio
    async def test_is_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("content")
        env = LocalEnvironment()
        assert await env.is_file(str(f)) is True
        assert await env.is_file(str(tmp_path)) is False

    @pytest.mark.asyncio
    async def test_mkdir(self, tmp_path):
        d = tmp_path / "new" / "dir"
        env = LocalEnvironment()
        await env.mkdir(str(d))
        assert d.is_dir()

    @pytest.mark.asyncio
    async def test_exec_shell(self):
        env = LocalEnvironment()
        result = await env.exec_shell("echo hello")
        assert result.stdout.strip() == "hello"
        assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_exec_shell_with_env(self):
        env = LocalEnvironment()
        result = await env.exec_shell(
            "echo $TEST_VAR",
            env={**os.environ, "TEST_VAR": "myvalue"},
        )
        assert "myvalue" in result.stdout

    @pytest.mark.asyncio
    async def test_exec_shell_timeout(self):
        env = LocalEnvironment()
        result = await env.exec_shell("sleep 10", timeout=1)
        assert result.returncode == -1
        assert "timed out" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_exec_shell_error(self):
        env = LocalEnvironment()
        result = await env.exec_shell("exit 42")
        assert result.returncode == 42

    @pytest.mark.asyncio
    async def test_glob(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "c.txt").write_text("c")
        env = LocalEnvironment()
        results = await env.glob("*.py", str(tmp_path))
        assert "a.py" in results
        assert "b.py" in results
        assert "c.txt" not in results

    @pytest.mark.asyncio
    async def test_list_dir(self, tmp_path):
        (tmp_path / "x.txt").write_text("x")
        (tmp_path / "y.txt").write_text("y")
        env = LocalEnvironment()
        entries = await env.list_dir(str(tmp_path))
        assert "x.txt" in entries
        assert "y.txt" in entries

    @pytest.mark.asyncio
    async def test_start_stop_noop(self):
        env = LocalEnvironment()
        await env.start()  # should not raise
        await env.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_implements_protocol(self):
        env = LocalEnvironment()
        assert isinstance(env, ExecutionEnvironment)

    @pytest.mark.asyncio
    async def test_shell_result_output(self):
        result = ShellResult(stdout="out", stderr="err", returncode=1)
        assert "out" in result.output
        assert "STDERR:" in result.output
        assert "Exit code: 1" in result.output

    @pytest.mark.asyncio
    async def test_shell_result_empty(self):
        result = ShellResult(stdout="", stderr="", returncode=0)
        assert result.output == "(no output)"


# ================================================================== #
# DockerEnvironment (unit tests -- no Docker daemon required)
# ================================================================== #


class TestDockerEnvironmentUnit:
    def test_not_running_initially(self):
        env = DockerEnvironment(image="test:latest")
        assert env.is_running is False
        assert env.container_id is None

    @pytest.mark.asyncio
    async def test_read_without_start_raises(self):
        env = DockerEnvironment()
        with pytest.raises(RuntimeError, match="not running"):
            await env.read_file("/workspace/test.txt")

    @pytest.mark.asyncio
    async def test_write_without_start_raises(self):
        env = DockerEnvironment()
        with pytest.raises(RuntimeError, match="not running"):
            await env.write_file("/workspace/test.txt", "content")

    @pytest.mark.asyncio
    async def test_exec_without_start_raises(self):
        env = DockerEnvironment()
        with pytest.raises(RuntimeError, match="not running"):
            await env.exec_shell("echo hi")

    def test_quote_method(self):
        # shlex.quote wraps in single quotes
        assert DockerEnvironment._quote("hello world") == "'hello world'"
        assert DockerEnvironment._quote("simple") == "simple"

    def test_custom_image_and_workspace(self):
        env = DockerEnvironment(image="node:18", workspace="/app", name="test-container")
        assert env._image == "node:18"
        assert env._workspace == "/app"
        assert env._name == "test-container"


# ================================================================== #
# Tool integration: set_environment / get_environment
# ================================================================== #


class TestToolEnvironmentIntegration:
    def test_default_is_local(self):
        env = get_environment()
        assert isinstance(env, LocalEnvironment)

    def test_set_and_get_environment(self):
        original = get_environment()
        try:
            new_env = LocalEnvironment()
            set_environment(new_env)
            assert get_environment() is new_env
        finally:
            set_environment(original)

    @pytest.mark.asyncio
    async def test_tools_use_environment(self, tmp_path):
        """Verify that tool functions actually use _environment."""
        from attractor_agent.tools.core import (
            _read_file,
            _write_file,
            set_allowed_roots,
        )

        set_allowed_roots([str(tmp_path)])
        try:
            # Write via tool
            f = tmp_path / "env_test.txt"
            await _write_file(str(f), "environment works")

            # Read via tool
            content = await _read_file(str(f))
            assert "environment works" in content
        finally:
            set_allowed_roots([os.getcwd()])
