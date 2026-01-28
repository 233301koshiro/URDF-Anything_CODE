#tripoのapiを呼び出して雪だるまのglbファイルを生成する
import asyncio
import os
import requests
from tripo3d import TripoClient
# エラー回避のため、TaskStatusをインポートできなくても動くようにする
try:
    from tripo3d.enums import TaskStatus
except ImportError:
    TaskStatus = None
def check_api_balance():
    url = "https://api.tripo3d.ai/v2/openapi/user/balance"
    headers = {
        "Authorization": f"Bearer {TRIPO_API_KEY}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # エラーならここで止まる
        
        data = response.json()
        # APIのレスポンス形式に合わせて表示
        # data['data']['balance'] に入っていることが多いです
        balance_info = data.get('data', {})
        print("--------------------------------")
        print(f"💰 API Wallet Balance: {balance_info.get('balance', 'Unknown')}")
        print(f"❄️ Frozen (使用中):    {balance_info.get('frozen', 'Unknown')}")
        print("--------------------------------")
        
    except Exception as e:
        print(f"確認失敗: {e}")
        if response.status_code == 403 or response.status_code == 401:
            print("→ APIキーが間違っているか、権限がありません。")

# 【ここにAPIキーを入れてください】
TRIPO_API_KEY = "tsk_S2g0SX4eTh3UCwlP7YqWHXW9lzwDhe-i57nOYZ3h2T7"

async def main():
    check_api_balance()
    # 既存の output フォルダがあれば掃除（任意）
    os.makedirs("output", exist_ok=True)

    print("🚀 Tripoに雪だるまの生成を依頼しています...")
    
    async with TripoClient(api_key=TRIPO_API_KEY) as client:
        # 1. 生成タスク開始
        task_id = await client.text_to_model(
            prompt="A simple snowman with two stacked snowballs, minimalist style",
            negative_prompt="low quality, blurry, complex details",
        )
        print(f"✅ タスク開始: {task_id}")

        # 2. 完了待機 (verbose=Trueで進捗が見れます)
        task = await client.wait_for_task(task_id, verbose=True)

        if task.status == TaskStatus.SUCCESS:
            print("🎉 生成完了！ ダウンロードします...")
            
            # 3. ダウンロード (./output フォルダに保存)
            files = await client.download_task_models(task, "./output")
            
            for model_type, path in files.items():
                print(f"📥 Downloaded {model_type}: {path}")
                
            # Step 2のためにファイル名を固定しておくと便利です
            # 通常 model.glb という名前で落ちてくることが多いですが、念のため確認
            # (ここでは何もしませんが、次のステップでフォルダ内のGLBを探します)
        else:
            print(f"❌ 生成失敗: {task.status}")

if __name__ == "__main__":
    asyncio.run(main())