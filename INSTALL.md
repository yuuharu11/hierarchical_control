## uvのインストール
```bash
apt update
apt install -y curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## 依存関係のインストール
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
apt install -y libgl1-mesa-glx
apt install -y libglib2.0-0
```

## LIBEROをEGLで動かすセットアップ

### なぜEGLで動くようになるのか
- MuJoCoのヘッドレス描画はEGLバックエンドを使います。
- EGLはGLVNDのvendor設定を参照し、どの実装(Mesa/NVIDIA)を使うか決めます。
- コンテナ内でMesa側だけが見えていると、`Cannot initialize a EGL device display` が起きることがあります。
- NVIDIAのEGL実装を明示し、`MUJOCO_GL=egl` と `PYOPENGL_PLATFORM=egl` を揃えると、GPU上でオフスクリーンレンダリングできるようになります。

### 追加で必要なOSライブラリ
```bash
apt update
apt install -y --no-install-recommends \
	libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 libegl1
```

### NVIDIA EGL vendor設定(コンテナ内)
```bash
mkdir -p /usr/share/glvnd/egl_vendor.d
cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
	"file_format_version": "1.0.0",
	"ICD": {
		"library_path": "libEGL_nvidia.so.0"
	}
}
EOF
```

### 環境変数の設定
```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```
