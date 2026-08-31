import os
import yaml
import time
import pandas as pd
from tqdm import tqdm  # 需要安装: pip install tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ================= 🔧 配置区 =================

load_dotenv(override=True)

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_DIR, "configs", "models.yaml")
PROMPTS_PATH = os.path.join(BASE_DIR, "configs", "prompts.yaml")
# Canonical human baseline (Phase 0: was data/processed/human_data.csv)
INPUT_FILE = os.path.join(BASE_DIR, "data", "interim", "human_data_full.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs", "corpus_final")  # 最终语料库目录

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 🛠️ 客户端工厂 =================


def get_client(provider_name):
    """获取客户端 (依赖系统全局代理)"""
    if provider_name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("❌ 缺少 OPENROUTER_API_KEY")
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/Anonymous/LLM_Bias_Research",
                "X-Title": "LLM Cultural Bias Analysis",
            },
            timeout=60.0,  # 增加超时时间，防止网络波动
        )
    elif provider_name == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        return OpenAI(
            api_key=api_key, base_url="https://api.deepseek.com", timeout=60.0
        )
    elif provider_name == "aliyun":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=60.0,
        )
    else:
        raise ValueError(f"❌ 未知 Provider: {provider_name}")


# ================= 🚀 全量生成主程序 =================


def run_production_generation():
    print(f"🚀 启动全量语料生成 (Production Mode)...")
    print(f"📂 输入文件: {INPUT_FILE}")
    print(f"📂 输出目录: {OUTPUT_DIR}")

    # 1. 加载配置
    with open(MODELS_PATH, "r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts_cfg = yaml.safe_load(f)

    # 2. 加载数据
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 human baseline: {INPUT_FILE}")
        return
    df = pd.read_csv(INPUT_FILE)
    # Canonical columns are id/region; tolerate legacy text_id/nationality.
    id_col = "id" if "id" in df.columns else "text_id"
    region_col = "region" if "region" in df.columns else "nationality"
    total_samples = len(df)
    print(f"📊 数据集总量: {total_samples} 条")

    # 3. 遍历模型 (根据 models.yaml 的注释情况，自动跳过被注释的模型)
    for model_cfg in models_cfg["models"]:
        display_name = model_cfg["display_name"]
        model_id = model_cfg["name"]
        provider = model_cfg["provider"]

        print(f"\n🔵 [{display_name}] 准备生成 ({provider})...")

        # 定义输出文件路径 (每个模型一个独立文件)
        output_file_path = os.path.join(
            OUTPUT_DIR, f"Corpus_{display_name.replace(' ', '_')}.csv"
        )

        # --- 🔄 断点续传逻辑 ---
        processed_ids = set()
        if os.path.exists(output_file_path):
            try:
                # 读取已生成的数据，获取 ID
                existing_df = pd.read_csv(output_file_path)
                if "source_id" in existing_df.columns:
                    processed_ids = set(existing_df["source_id"].astype(str))
                print(
                    f"   🔄 检测到历史记录: 已完成 {len(processed_ids)}/{total_samples} 条。将继续生成..."
                )
            except Exception as e:
                print(f"   ⚠️ 读取历史文件失败，可能文件损坏: {e}")
        else:
            print(f"   ✨ 新建文件，开始从头生成...")
            # 初始化 CSV 头
            pd.DataFrame(
                columns=[
                    "source_id",
                    "region",
                    "topic",
                    "generated_text",
                    "model_id",
                    "latency",
                ]
            ).to_csv(output_file_path, index=False)

        # 获取客户端
        try:
            client = get_client(provider)
        except Exception as e:
            print(f"   ❌ 客户端初始化失败: {e} -> 跳过此模型")
            continue

        # --- 📝 逐条处理 ---
        # 使用 tqdm 显示进度条
        pbar = tqdm(
            total=total_samples, desc=f"   Generating {display_name}", unit="doc"
        )

        # 更新进度条到已完成的位置
        pbar.update(len(processed_ids))

        for index, row in df.iterrows():
            source_id = str(row[id_col])

            # 如果已经跑过，跳过
            if source_id in processed_ids:
                continue

            country_code = row[region_col]
            topic = row["topic"]
            country_name = prompts_cfg["persona_mapping"].get(
                country_code, country_code
            )

            # 构建 Prompt (极简同构逻辑)
            if country_code == "ENS":
                identity_block = prompts_cfg["identity_definitions"]["Native"]
            else:
                identity_block = prompts_cfg["identity_definitions"]["Learner"].format(
                    country=country_name
                )

            sys_msg = prompts_cfg["universal_prompt"]["system"].format(
                identity_block=identity_block
            )
            usr_msg = prompts_cfg["universal_prompt"]["user"].format(topic=topic)

            # API 调用 (带重试机制)
            max_retries = 3
            content = ""
            duration = 0

            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": usr_msg},
                        ],
                        temperature=0.7,
                        max_tokens=600,
                    )
                    duration = time.time() - start_time
                    content = response.choices[0].message.content.strip()
                    break  # 成功则跳出重试循环
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)  # 失败等待 2 秒重试
                    else:
                        print(f"\n   ❌ Sample {source_id} 失败: {e}")
                        content = "[ERROR]"  # 标记错误

            # 实时追加写入 CSV (防止程序崩溃数据全丢)
            new_row = pd.DataFrame(
                [
                    {
                        "source_id": source_id,
                        "region": country_code,
                        "topic": topic,
                        "generated_text": content,
                        "model_id": model_id,
                        "latency": round(duration, 2),
                    }
                ]
            )
            new_row.to_csv(output_file_path, mode="a", header=False, index=False)

            pbar.update(1)  # 更新进度条

        pbar.close()
        print(f"✅ {display_name} 生成完毕！\n")

    print(f"🎉 所有任务结束。请检查 {OUTPUT_DIR}")


if __name__ == "__main__":
    run_production_generation()
