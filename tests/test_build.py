"""构建契约（A0 第 3 行）：C 核是安装期编好的包内共享库，导入时不再现编。

断言内容（SESSION 卡片「交什么」第 5 条）：
1. `load()` 装载的共享库位于已安装的 xmf 包目录内（与 `_native.py` 同层），
   产物名随平台（Windows: xmf_core.dll / Linux: libxmf_core.so）；
   库文件不得来自仓库源树 src/，也不得来自临时目录（临时现编的落点）。
2. `xmf._native` 没有 `_compile`，模块源码中无 subprocess / gcc 痕迹。
3. 装载出的库上有四步 C 符号 xmf_setup / xmf_step / xmf_fields / xmf_finalize。

只测构建契约本身，不靠整条激波管「看起来像」；不读 pluto.ini。
"""
from __future__ import annotations

import ctypes
import inspect
import sys
import tempfile
from pathlib import Path

import xmf
from xmf import _native

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_TREE = REPO_ROOT / "src"
TMP_ROOT = Path(tempfile.gettempdir()).resolve()

_LIB_NAME = "xmf_core.dll" if sys.platform == "win32" else "libxmf_core.so"
_FOUR_STEP = ("xmf_setup", "xmf_step", "xmf_fields", "xmf_finalize")


def _package_dir() -> Path:
    """已安装 xmf 包目录（_native.py 所在目录）。"""
    return Path(_native.__file__).resolve().parent


def _loaded_lib_path() -> Path:
    """load() 实际装载的共享库路径（跨平台）。

    复用 load()，不自己另起 CDLL。ctypes.CDLL 把装载用的名字存在 `_name`；
    _native.load() 传入的是包内产物的已解析绝对路径，故 Windows / POSIX 上
    `_name` 即库文件路径。
    """
    lib = _native.load()
    assert isinstance(lib, ctypes.CDLL), (
        f"load() 必须仍返回 ctypes.CDLL，实际得到 {type(lib)!r}"
    )
    return Path(lib._name).resolve()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def test_lib_lives_inside_installed_package_dir():
    """库文件必须与 _native.py 同层，且产物名符合平台约定。"""
    pkg_dir = _package_dir()
    lib_path = _loaded_lib_path()
    assert lib_path.is_file(), f"装载的库文件不存在: {lib_path}"
    assert lib_path.name == _LIB_NAME, (
        f"本平台产物名应为 {_LIB_NAME}，实际为 {lib_path.name}"
    )
    assert lib_path.parent == pkg_dir, (
        f"共享库必须位于已安装包目录 {pkg_dir} 内（与 _native.py 同层），"
        f"实际在 {lib_path}"
    )


def test_lib_not_from_source_tree_or_tempdir():
    """库不得来自源树 src/（旧内环产物落点）或临时目录（现编落点）。"""
    lib_path = _loaded_lib_path()
    assert not _is_under(lib_path, SRC_TREE), (
        f"装载的库不得来自仓库源树 src/: {lib_path}"
    )
    assert not _is_under(lib_path, TMP_ROOT), (
        f"装载的库不得来自临时目录（导入时现编的迹象）: {lib_path}"
    )


def test_imported_xmf_is_installed_package_not_source_tree():
    """import xmf 解析到的应是已安装包，防止 pytest 从仓库根误吃源树。"""
    init_path = Path(xmf.__file__).resolve()
    assert not _is_under(init_path, SRC_TREE), (
        f"import xmf 解析进了仓库源树: {init_path}"
    )


def test_native_has_no_compile_hook():
    """_native 不得保留任何导入时编译钩子。"""
    assert not hasattr(_native, "_compile"), (
        "xmf._native 仍带 _compile（导入时现编钩子未删净）"
    )


def test_native_source_free_of_subprocess_and_gcc():
    """_native 源码中不得出现 subprocess / gcc 调用痕迹（构建已移交后端）。"""
    source = inspect.getsource(_native)
    assert "subprocess" not in source, (
        "xmf._native 源码中出现 subprocess（禁止导入时起子进程编译）"
    )
    assert "gcc" not in source.lower(), (
        "xmf._native 源码中出现 gcc 字样（禁止残留工具链调用）"
    )


def test_four_step_symbols_present():
    """装载出的库上四步 C 符号齐全（getattr 不抛异常）。"""
    lib = _native.load()
    for name in _FOUR_STEP:
        try:
            getattr(lib, name)
        except AttributeError:
            raise AssertionError(f"装载的库缺少 C 符号 {name!r}") from None
