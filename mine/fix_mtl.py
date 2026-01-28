import os

# 出力フォルダ
OUTPUT_DIR = "output/gemini"
OBJ_NAME = "check_me.obj"

# 画像ファイルを探す
texture_file = None
if os.path.exists(OUTPUT_DIR):
    files = os.listdir(OUTPUT_DIR)
    for f in files:
        if f.endswith(".png"):
            texture_file = f
            break

if not texture_file:
    print(f"エラー: {OUTPUT_DIR} フォルダ内に .png ファイルが見つかりません。")
    exit()

obj_path = os.path.join(OUTPUT_DIR, OBJ_NAME)
mtl_name = "material.mtl"
mtl_path = os.path.join(OUTPUT_DIR, mtl_name)

# ==========================================
# ★ここが変更点: Ke (自己発光) を追加して明るくする
# ==========================================
mtl_content = f"""newmtl material_0
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.000 0.000 0.000
Ke 1.000 1.000 1.000
map_Kd {texture_file}
"""

with open(mtl_path, "w") as f:
    f.write(mtl_content)
print(f"マテリアルを更新しました（明るさUP）: {mtl_path}")

# OBJファイルへの紐付け確認（念のため再実行）
if os.path.exists(obj_path):
    with open(obj_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    has_mtllib = False
    has_usemtl = False
    faces_started = False

    if any("mtllib" in l for l in lines): has_mtllib = True
    if any("usemtl" in l for l in lines): has_usemtl = True

    if not has_mtllib:
        new_lines.append(f"mtllib {mtl_name}\n")

    for line in lines:
        if line.startswith("f ") and not faces_started:
            faces_started = True
            if not has_usemtl:
                new_lines.append("usemtl material_0\n")
        new_lines.append(line)

    with open(obj_path, "w") as f:
        f.writelines(new_lines)
    print(f"OBJファイルも確認済み: {obj_path}")

print("完了！Choreonoidで再読み込みしてください。")