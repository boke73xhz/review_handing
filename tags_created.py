# -*- coding: utf-8 -*-
"""
pipeline.py  —  话题标签自动化管道（通用版）
==============================================
输入: reviews_lag.csv（仅需 review + lag_fa 列）
流程:
  1. 首次运行: 分词统计 → keyword_frequencies.txt
  2. 根据高频词创建 topic_seeds.json（手动）
  3. 再次运行: Word2Vec扩展 → 标注 → 报告
==============================================
"""
import os, re, json
from collections import Counter
import numpy as np
import pandas as pd
import jieba
from gensim.models import Word2Vec

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(ROOT, exist_ok=True)

OUT_DIR = os.path.join(ROOT, "tags_created")
os.makedirs(OUT_DIR, exist_ok=True)

INPUT          = os.path.join(ROOT, "lang_cla", "review_lang.csv")
SEEDS_FILE     = os.path.join(ROOT, "tags_created", "topic_seeds.json")
OUT_FREQ       = os.path.join(OUT_DIR, "keyword_frequencies.txt")
OUT_ALLWORDS   = os.path.join(OUT_DIR, "all_words.txt")
OUT_PROMPT     = os.path.join(OUT_DIR, "seed_prompt.txt")
OUT_LABELS     = os.path.join(OUT_DIR, "labels.json")
OUT_DETAIL     = os.path.join(OUT_DIR, "label_detail.txt")
OUT_TAGGED     = os.path.join(OUT_DIR, "reviews_tagged.csv")
OUT_REPORT     = os.path.join(OUT_DIR, "coverage_report.txt")
MODEL_PATH     = os.path.join(OUT_DIR, "word2vec.model")
SIM_THRESHOLD  = 0.55

# ============================================================
# 停用词（中英分开）
# ============================================================
STOPWORDS_ZH = {
    '的','人','是','在','我','有','和','就','不','了','都','一','一个','中','也','很',
    '到','说','要','去','你','会','着','没有','看','好','自己','这','他','她','它','那',
    '把','可','能','做','被','让','但','我们','可以','这个','那个','什么','怎么','如果',
    '因为','所以','但是','然后','之后','已经','需要','应该','觉得','比较','非常','真的',
    '还是','只是','就是','的话','现在','感觉','可能','一直','一点','有点','不过','不能',
    '不会','不是','很多','这些','那些','这样','那样','这种','那种','一些','以为','出来',
    '起来','过来','进去','出','来','太','小','大','少','没','挺','特别','最','更','越',
    '还','再','才','只','刚','正','游戏','还有','只能','直接','体验','目前','知道','小时',
    '希望','建议','时间','一样','不少','完全','一直','开始','主要','找到','东西','地方',
    '应该','比较','非常','真的','一点','有点','不过','不会','不是','很多','已经','需要',
    '之后','可以','但是','这个','那个','因为','所以','这样','那样','感觉','可能','现在',
    '觉得','什么','如果','怎么','然后','那种','这种','还是','就是','只是','的话','一下',
    '想要','毕竟','发现','方面','等等','整个','来说','确实','那个','每个','之间','没有',
    '游戏',
}

STOPWORDS_EN = {
    'the','is','it','to','and','of','a','in','that','for','this','with',
    'be','not','but','on','as','are','i','my','me','you','your','we','so','if',
    'no','its','very','all','just','like','can','get','has','more','don','got',
    'play','playing','played','time','hours','hour','game','games',
    'review','early','access','steam','one','really','still','even','would',
    'much','good','bad','fun','make','made','well','also','great','love','hate',
    'have','there','some','an','at','do','or','building','was','had','up',
    'were','been','will','am','did','about','being','each','few','ve','re','ll','s','t',
    'they','them','their','lot','out','what','far','see','only','things','how','from',
    'feels','when','into','off','back','than','over','through','after','before',
    'need','going','get','got','think','know','go','say','said','take','put',
    'any','way','thing','here','down','let','want','really','still','even','try',
    # 拉丁语系停用词（西/葡/法/意）
    'de','la','que','en','el','los','se','es','un','una','lo','le',
    'mejor','para','muy','por','con','mas','del',
    'que','como','está','están','fue','son','era','han','hay','sus',
    'las','nos','tu','mis','te','ya','si','ni','tan','cada','todo',
    'dos','entre','desde','sobre','donde','cuando','porque','pero',
    'é','ao','uma','com','não','sim','estou','está','são','foi',
    'das','dos','na','no','mais','menos','outro','outra','pelo','pela',
    'dans','avec','pour','sur','sous','tout','tous','toute','toutes',
    'pas','plus','bien','bon','très','leur','leurs','aux',
    'der','die','das','ein','eine','einen','einem','einer',
    'ist','und','von','zu','mit','sich','auf','für','nicht',
    'auch','aber','oder','im','am','zum','zur','den','dem',
    'des','war','hat','wird','kann','soll','muss',
    'und','die','der','das','in','zu','von','mit',
    'и','в','не','на','с','по','из','от','для','у','о',
    'что','как','это','так','все','его','ее','их','мой',
    'di','che','non','una','sono','stato','fatto','essere',
    'het','een','van','op','dat','zijn','niet','aan',
}

CHINESE_LANGS = {'schinese', 'tchinese', 'chinese'}

def _is_chinese_lang(lang):
    """根据 Steam language 字段判断是否中文"""
    if lang is None:
        return None  # 无法判断
    return str(lang).lower() in CHINESE_LANGS

def tokenize(text, lang=None):
    """根据 language 列选择分词器：中文用 jieba，其余用空格
    如果 lang 为 None，fallback 到 CJK 字符检测"""
    text = re.sub(r'\[[^\]]*\]|<[^>]*>|[^\w\s]', ' ', str(text))
    is_cn = _is_chinese_lang(lang)
    if is_cn is None:
        # fallback: 检测文本是否含汉字
        is_cn = any('\u4e00' <= ch <= '\u9fff' for ch in text)
    if is_cn:
        words = jieba.cut(text)
        stopwords = STOPWORDS_ZH
    else:
        words = text.lower().split()
        stopwords = STOPWORDS_EN
    result = []
    for w in words:
        w = w.strip()
        if len(w) < 2: continue
        if w.isdigit(): continue
        if w in stopwords: continue
        result.append(w)
    return result


# ============================================================
def step1_freq():
    """分词统计，输出高频词 + 全量词表"""
    print("=" * 55)
    print("[Step 1] 分词统计")
    print("=" * 55)

    df = pd.read_csv(INPUT)
    print(f"  评论数: {len(df)}")

    sentences = []
    counter = Counter()
    for _, row in df.iterrows():
        words = tokenize(row["review"], row.get("language"))
        if len(words) >= 2:
            sentences.append(words)
            counter.update(set(words))

    print(f"  训练语料 (>=2词): {len(sentences)} 条")
    print(f"  不重复词数: {len(counter)}")
    # 语系分布
    if 'language' in df.columns:
        lang_dist = df['language'].value_counts()
        print(f"  语系分布: {dict(lang_dist)}")

    # 高频词
    with open(OUT_FREQ, "w", encoding="utf-8") as f:
        f.write(f"## 分词统计  —  {len(df)} 条评论\n\n")
        f.write(f"{'关键词':<18} {'频次':>6}  {'覆盖率':>8}\n")
        f.write("-" * 40 + "\n")
        for w, freq in counter.most_common(80):
            f.write(f"{w:<18} {freq:>6}  {freq/len(df)*100:>7.1f}%\n")
    print(f"  -> {OUT_FREQ}")

    # 全量词表
    with open(OUT_ALLWORDS, "w", encoding="utf-8") as f:
        f.write(f"# 全量词表  —  {len(counter)} 个词\n")
        f.write(f"{'word':<18} {'doc_freq':>8}\n")
        for w, freq in counter.most_common():
            f.write(f"{w:<18} {freq:>8}\n")
    print(f"  -> {OUT_ALLWORDS}")

    return df, sentences, counter


# ============================================================
def step2_train_and_expand(sentences, counter, topic_seeds):
    """Word2Vec 训练 + 词扩展"""
    print("\n" + "=" * 55)
    print("[Step 2] Word2Vec 训练 + 词扩展")
    print("=" * 55)

    print(f"  主题数: {len(topic_seeds)}")
    total_seeds = sum(len(v) for v in topic_seeds.values())
    print(f"  种子词数: {total_seeds}")

    # 训练
    print("  训练 Word2Vec...")
    model = Word2Vec(sentences, vector_size=150, window=5, min_count=2,
                     workers=4, epochs=50, sg=1)
    model.save(MODEL_PATH)
    print(f"  词表: {len(model.wv)} 词, 维度: {model.wv.vector_size}")

    # 种子词白名单
    seed_whitelist = set()
    for words in topic_seeds.values():
        for w in words:
            seed_whitelist.add(w.lower())

    # 主题中心向量
    topic_vecs = {}
    for topic, words in topic_seeds.items():
        vecs = [model.wv[w] for w in words if w in model.wv]
        if vecs:
            topic_vecs[topic] = np.mean(vecs, axis=0)
        else:
            print(f"  ! 主题 [{topic}] 没有种子词在词表中")

    print(f"  有向量的主题: {len(topic_vecs)}/{len(topic_seeds)}")

    # 候选词
    all_words = set()
    for w, f in counter.items():
        if f >= 2:
            all_words.add(w)
    for words in topic_seeds.values():
        all_words.update(words)
    print(f"  待映射词: {len(all_words)} 个")

    # 映射
    topic_map = {t: [] for t in topic_seeds}
    matched = 0
    all_best_sims = []

    for word in all_words:
        wl = word.lower()
        if word not in model.wv: continue
        if word.isascii() and wl not in seed_whitelist: continue
        wv = model.wv[word]
        best_sim = 0.0
        best_topic = None
        for topic, tv in topic_vecs.items():
            sim = float(np.dot(wv, tv) / (np.linalg.norm(wv) * np.linalg.norm(tv)))
            if sim > best_sim:
                best_sim = sim
                best_topic = topic
        all_best_sims.append(best_sim)
        if best_sim >= SIM_THRESHOLD and best_topic:
            topic_map[best_topic].append((word, round(best_sim, 3)))
            matched += 1

    for t in topic_map:
        topic_map[t].sort(key=lambda x: -x[1])

    # 种子词强制加入
    for topic, words in topic_seeds.items():
        existing = {x[0] for x in topic_map[topic]}
        for w in words:
            if w not in existing and w in model.wv:
                topic_map[topic].insert(0, (w, 1.0))

    if all_best_sims:
        arr = np.array(all_best_sims)
        print(f"  相似度: min={arr.min():.3f} p25={np.percentile(arr,25):.3f} median={np.median(arr):.3f} p75={np.percentile(arr,75):.3f} max={arr.max():.3f}")
        for t in [0.4, 0.5, 0.55, 0.6]:
            n = (arr >= t).sum()
            print(f"    阈值{t}: {n}/{len(arr)} ({n/len(arr)*100:.1f}%)")
    print(f"  映射成功: {matched} 个扩展词 (阈值 {SIM_THRESHOLD})")

    # 保存 labels.json
    labels_out = {}
    for t, wordlist in topic_map.items():
        labels_out[t] = [{"word": w, "sim": s} for w, s in wordlist]
    with open(OUT_LABELS, "w", encoding="utf-8") as f:
        json.dump(labels_out, f, ensure_ascii=False, indent=2)
    total_expanded = sum(len(v) for v in topic_map.values())
    print(f"  -> {OUT_LABELS} ({total_expanded} 词条)")

    return topic_map, model


# ============================================================
def step3_tag(df, topic_map):
    """标注评论"""
    print("\n" + "=" * 55)
    print("[Step 3] 标注评论")
    print("=" * 55)

    word_to_topics = {}
    for topic, wordlist in topic_map.items():
        for w, sim in wordlist:
            word_to_topics.setdefault(w.lower().strip(), []).append(topic)

    tagged = 0
    all_tags = []
    tag_counts = Counter()

    for _, row in df.iterrows():
        words = tokenize(row["review"], row.get("language"))
        word_set = set(w.lower().strip() for w in words)
        topics = set()
        for w in word_set:
            if w in word_to_topics:
                topics.update(word_to_topics[w])
        tag_str = "|".join(sorted(topics)) if topics else ""
        all_tags.append(tag_str)
        if topics:
            tagged += 1
            for t in topics:
                tag_counts[t] += 1

    df["tags"] = all_tags
    total = len(df)
    df.to_csv(OUT_TAGGED, index=False, encoding="utf-8-sig")
    print(f"  -> {OUT_TAGGED}")
    print(f"  标注: {tagged}/{total} ({tagged/total*100:.1f}%)")
    print(f"  未标注: {total - tagged} ({(total - tagged)/total*100:.1f}%)")

    return tagged, tag_counts, total


# ============================================================
def step4_report(topic_map, topic_seeds, tagged, tag_counts, total):
    """生成报告 + 详细标签"""
    print("\n" + "=" * 55)
    print("[Step 4] 生成报告")
    print("=" * 55)

    total_expanded = sum(len(v) for v in topic_map.values())

    # --- 覆盖率报告 ---
    lines = []
    lines.append("=" * 60)
    lines.append("  话题标签映射 & 评论标注  —  覆盖率报告")
    lines.append("=" * 60)
    lines.append(f"\n  总评论: {total}")
    lines.append(f"  已标注: {tagged} ({tagged/total*100:.1f}%)")
    lines.append(f"  未标注: {total - tagged} ({(total - tagged)/total*100:.1f}%)")
    lines.append(f"\n  主题数: {len(topic_map)}")
    lines.append(f"  标签映射词总量: {total_expanded}")
    lines.append(f"  Word2Vec 相似度阈值: {SIM_THRESHOLD}")
    lines.append(f"\n  {'标签':<18} {'种子':>5} {'扩展':>5} {'标注评论':>8}")
    lines.append("  " + "-" * 45)

    for t in sorted(topic_map, key=lambda k: -tag_counts.get(k, 0)):
        n_seed = len(topic_seeds.get(t, []))
        n_extra = len(topic_map[t]) - n_seed
        n_tagged = tag_counts.get(t, 0)
        lines.append(f"  {t:<18} {n_seed:>5} {n_extra:>5} {n_tagged:>8}")

    lines.append("\n  " + "-" * 45)
    lines.append(f"\n  详细映射: labels.json / label_detail.txt")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  -> {OUT_REPORT}")

    # --- 详细标签报告 ---
    detail = []
    detail.append("=" * 72)
    detail.append("  话题标签详细报告 — 种子词 & Word2Vec 扩展词")
    detail.append("=" * 72)
    detail.append("")
    detail.append("  说明:")
    detail.append("    ★ 种子词  (sim = 1.000) — 人工定义的核心词")
    detail.append("    · 扩展词  (sim < 1.000) — Word2Vec 余弦相似度映射")
    detail.append(f"    阈值: {SIM_THRESHOLD}")
    detail.append("")
    detail.append("=" * 72)

    for topic in topic_map:
        wordlist = topic_map[topic]
        seeds = [(w, s) for w, s in wordlist if s >= 0.999]
        expanded = [(w, s) for w, s in wordlist if s < 0.999]
        detail.append("")
        detail.append(f"  [{topic}]  种子:{len(seeds)} | 扩展:{len(expanded)} | 合计:{len(wordlist)}")
        detail.append("  " + "-" * 68)
        detail.append("  [种子词]")
        for w, sim in seeds:
            detail.append(f"    ★ {w}")
        detail.append("  [扩展词] (按相似度降序)")
        for w, sim in expanded:
            detail.append(f"    · {w:<20} {sim:.3f}")

    detail.append("")
    detail.append("=" * 72)
    detail.append(f"  总计: {len(topic_map)} 个主题, {total_expanded} 个词条")
    detail.append("=" * 72)

    with open(OUT_DETAIL, "w", encoding="utf-8") as f:
        f.write("\n".join(detail))
    print(f"  -> {OUT_DETAIL}")

    # 打印报告
    print("\n".join(lines))


# ============================================================
def generate_seed_prompt(counter, total_comments):
    """生成可用于 LLM 的种子词创建 prompt"""
    top_words = counter.most_common(80)
    freq_lines = []
    for i, (w, f) in enumerate(top_words, 1):
        freq_lines.append(f"  {i:>2}. {w:<14} {f}/{total_comments} ({f/total_comments*100:.1f}%)")

    prompt = f'''你是一位游戏评论数据分析专家。以下是从 {total_comments} 条 Steam 游戏评论中提取的 80 个高频关键词（按文档频率排序）。数据包含中英文评论，关键词已统一经过分词处理（中文jieba分词，英文空格分词）。

## 高频关键词
{"".join(chr(10) for _ in range(1))}
{chr(10).join(freq_lines)}

## 任务
根据以上高频词，归纳出 8-15 个"话题标签"，并为每个标签选取 5-15 个"种子词"。
种子词必须来自高频词列表或列表中词的合理同义变体。
如果数据包含多种语言（中/英/西/葡/法/德/俄等），每个标签需覆盖主要语系的关键词。
输出严格 JSON，格式如下：

{{
  "标签名A": ["种子词1", "种子词2", ...],
  "标签名B": ["种子词3", "种子词4", ...]
}}

## 归并原则
1. 直接收录: 高频词本身即话题核心概念 → 作为标签名和种子词（如"火山"→火山/后期难度）
2. 横向聚合: 多个高频词指向同一话题 → 合并（如"内容"+"更新"+"前期"→内容与更新）
3. 反面归入: 高频词的反面也归入该话题（如"单人"→联机体验话题；"知道"→"不知道"→新手指引）
4. 具体化补词: 从高频词联想玩家常用口语表达补充为种子词（如"问题"→"卡死""闪退""崩溃"）
5. 容忍交叉: 同一词可出现在多个标签（如"boss"同时在难度标签和怪物标签）
6. 标签命名: 用"核心维度/具体方面"格式，如"优化/性能""建造/地形"
7. 双语覆盖: 同一标签同时收录中英文种子词（如"bug""卡死""crash"→Bug标签;"multiplayer""联机"→联机标签）

## 示例（游戏 Romestead 的 11 个标签，每个标签含中英文种子词）
{{
  "内容与更新":   ["内容","更新","期待","前期","重复","未来","玩法","流程","后期","EA","content","update","early access","unfinished"],
  "火山/后期难度": ["火山","boss","难度","沙漠","战斗","数值","强度","火焰","灰烬","volcano","lava","difficulty","ashlands"],
  "Bug与技术问题": ["bug","问题","卡顿","闪退","崩溃","卡死","消失","数据","丢失","crash","glitch","freeze","error","problema","erreur","fehler"],
  "联机体验":     ["联机","多人","单人","朋友","服务器","延迟","掉线","multiplayer","coop","co-op","server","lag","solo","juntos","juego online"],
  "建造与地形":   ["建造","建筑","地图","城墙","地形","围墙","位置","基地","build","construct","wall","terrain","base"],
  "NPC/村民系统": ["村民","NPC","居民","镇民","villager","townsfolk","citizen"],
  "物流/自动化":  ["自动化","自动","搬运","材料","物流","效率","资源","仓库","automation","logistic","transport","storage","supply"],
  "优化/性能":    ["优化","配置","帧数","性能","内存","显卡","GPU","fps","performance","optimization","lag","framerate"],
  "怪物/Boss设计":["怪物","敌人","小怪","精英","仇恨","刷怪","袭击","enemy","mob","boss fight","spawn","raid"],
  "新手指引":     ["引导","教程","指引","说明","讲解","提示","教学","tutorial","guide","new player","instruction"],
  "好评/推荐":    ["好玩","推荐","不错","值得","喜欢","上头","惊艳","出色","recommend","love","addictive","masterpiece","worth","bueno","genial","top","incroyable"]
}}

请基于当前游戏的高频词，输出 topic_seeds.json 的内容。确保每个标签同时包含中英文种子词。只输出 JSON，不要其他文字。'''

    with open(OUT_PROMPT, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"  -> {OUT_PROMPT}")
    return prompt


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    if not os.path.exists(INPUT):
        print(f"错误: 找不到 {INPUT}")
        print("请将评论 CSV 命名为 reviews_lag.csv 放入当前目录")
        exit(1)

    # 检查种子文件
    if not os.path.exists(SEEDS_FILE):
        print("=" * 55)
        print("  未找到 topic_seeds.json — 分词统计 + 生成 LLM prompt")
        print("=" * 55)
        df, sentences, counter = step1_freq()
        generate_seed_prompt(counter, len(df))
        print("\n" + "=" * 55)
        print("  下一步（二选一）:")
        print("")
        print("  [A] 用 LLM 生成:")
        print("    1. 复制 seed_prompt.txt 全部内容")
        print("    2. 粘贴到 ChatGPT/Claude/DeepSeek")
        print("    3. 将返回的 JSON 保存为 topic_seeds.json")
        print("")
        print("  [B] 手动创建:")
        print("    1. 查看 keyword_frequencies.txt 中的高频词")
        print("    2. 创建 topic_seeds.json，格式:")
        print('       {')
        print('         "标签名1": ["种子词A", "种子词B"],')
        print('         "标签名2": ["种子词C", "种子词D"]')
        print('       }')
        print("")
        print("  然后: python pipeline.py 完成标注")
        print("=" * 55)
        exit(0)

    # 加载种子词
    with open(SEEDS_FILE, encoding="utf-8") as f:
        topic_seeds = json.load(f)

    # 全流程
    df, sentences, counter = step1_freq()
    topic_map, model = step2_train_and_expand(sentences, counter, topic_seeds)
    tagged, tag_counts, total = step3_tag(df, topic_map)
    step4_report(topic_map, topic_seeds, tagged, tag_counts, total)

    print("\n" + "=" * 55)
    print(f"  全部输出: {ROOT}")
    print("=" * 55)
    print("完成!")
