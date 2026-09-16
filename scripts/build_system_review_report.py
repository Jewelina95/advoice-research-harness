"""Render a bounded review from real pilot artifacts, without rerunning models."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path


def metric_values(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)
                if row["condition"] == "Ours" and row["analysis_scope"] == "full_available_cohort"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--historical-artifacts", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--prepare-pilot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, details, records = [], [], []
    def link(path):
        return html.escape(os.path.relpath(path.resolve(), output.parent), quote=True)
    for path in sorted((args.workspace / "artifacts").glob("*/layer_a_metrics.csv")):
        name = path.parent.name
        new = metric_values(path)
        old_path = args.historical_artifacts / name / "layer_a_metrics.csv"
        old = metric_values(old_path) if old_path.exists() else {}
        delta = f'{new["accuracy"] - old["accuracy"]:+.3f}' if "accuracy" in old else "Not available"
        records.append({"dataset": name, "current": new, "historical": old,
                        "current_file": str(path.resolve()), "historical_file": str(old_path.resolve())})
        report = args.workspace / "reports/datasets" / name / "latest"
        rows.append(f'<tr><td>{html.escape(name)}</td><td>{int(new["n"])}</td>'
                    f'<td>{old.get("accuracy", float("nan")):.3f}</td><td>{new["accuracy"]:.3f}</td>'
                    f'<td>{delta}</td><td>{new["macro_auroc_ovr"]:.3f}</td>'
                    f'<td><a href="{link(report / "evaluation_report.html")}">Full evaluation</a></td></tr>')
        details.append(f'<section><h2>{html.escape(name)}</h2><p>Layer A: prediction and screening metrics. '
                       'Layer B: engineering and evidence checks; not a clinician trial.</p>'
                       f'<img loading="lazy" src="{link(report / "assets/layer_a_summary.png")}" alt="Layer A">'
                       f'<img loading="lazy" src="{link(report / "assets/layer_b_summary.png")}" alt="Layer B"></section>')
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    pilot = None
    pilot_section = ""
    if args.prepare_pilot:
        selection = json.loads((args.prepare_pilot / "results.json").read_text(encoding="utf-8"))
        pilot = json.loads((args.prepare_pilot / "test_evaluation/results.json").read_text(encoding="utf-8"))
        comparisons = [("ADvoice locked candidate", pilot["metrics"])]
        comparisons.extend((key.replace("_", " "), value["metrics"])
                           for key, value in pilot["released_comparisons"].items() if value["available"])
        cells = "".join(f'<tr><td>{html.escape(name)}</td><td>{values["accuracy"]:.4f}</td>'
                        f'<td>{values["macro_f1"]:.4f}</td><td>{values["macro_auc_ovr"]:.4f}</td>'
                        f'<td>{values["micro_auc_ovr"]:.4f}</td></tr>' for name, values in comparisons)
        matrix = pilot["metrics"]["confusion_matrix_HC_MCI_AD"]
        confusion = "".join(f'<tr><th>{label}</th>' + ''.join(f'<td>{value}</td>' for value in row) + '</tr>'
                            for label, row in zip(["HC", "MCI", "ADRD"], matrix))
        pilot_section = f'''<section><h2>PREPARE：锁定候选后的实际测试</h2>
<p>公开训练集 1,295 人、验证集 327 人。预先限定八个候选，依据验证集选择后冻结模型；
验证准确率为 {selection['winner']['accuracy']:.4f}。测试阶段没有重新训练或修改分类阈值。</p>
<p class="notice">候选在 412 人测试集上判对 {pilot['correct']} 人，没有超过历史独立模型，也没有超过 SpeechCARE。
本轮不把该候选升级为正式模型。该测试集已有历史查看记录，因此这是回顾性比较，不是全新确认性验证。</p>
<div class="table"><table><tr><th>Model / released checkpoint</th><th>Accuracy</th><th>Macro F1</th>
<th>Macro AUROC</th><th>Micro AUROC</th></tr>{cells}</table></div>
<h3>错误集中在哪里</h3><p>下表行是真实类别，列是预测类别。三分类中的 ADRD 对应本地代码的 AD 标签，
不代表每人都有阿尔茨海默病生物标志物确诊。</p><div class="table"><table>
<tr><th>True / predicted</th><th>HC</th><th>MCI</th><th>ADRD</th></tr>{confusion}</table></div>
<p>冻结表示上的新融合候选偏向 HC。验证阶段，同编码器下加入认知状态的堆叠模型多判对 5 人，
但直接拼接状态反而降低准确率。这说明状态信息的使用方式重要，却不能证明认知分支能稳定提高最终测试表现。</p>
<p>同一测试对象不等于同一输入和训练预算：本轮复用冻结声学、文本表示及本地转录，SpeechCARE 进行编码器训练。
这些是待检验的差异，不能直接断言是误差原因。应在开发集固定输入、预算和划分，分开验证任务状态、片段表示和 Agent 修正。</p>
<p><a href="{link(args.prepare_pilot / 'results.json')}">Validation record</a> ·
<a href="{link(args.prepare_pilot / 'test_evaluation/results.json')}">Locked test record</a> ·
<a href="https://www.nature.com/articles/s41746-025-02026-x">SpeechCARE paper</a></p></section>'''
    comparison = ''.join(f'<tr><td>{html.escape(key)}</td><td>{item["advoice"]:.4f}</td>'
                         f'<td>{item["speechcare_published_mean"]:.4f}</td><td>{item["delta"]:+.4f}</td></tr>'
                         for key, item in benchmark["comparison"].items())
    body = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ADvoice 系统审查与实测</title><style>
*{box-sizing:border-box}body{margin:0;color:#232c32;background:#fff;font:17px/1.7 system-ui}
header,main{max-width:1250px;margin:auto;padding:30px 24px}header{border-bottom:3px solid #167c80}
h1{font-size:30px;margin:12px 0}h2{font-size:23px;margin:0 0 14px}h3{font-size:18px}
section{padding:30px 0;border-bottom:1px solid #d7dfe1}p{max-width:1000px}a{color:#126e88}
.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:15px}
td,th{text-align:left;padding:12px;border-bottom:1px solid #d7dfe1;overflow-wrap:anywhere}
th{background:#edf4f4}img{display:block;width:100%;height:auto;margin:24px 0}
.notice{border-left:4px solid #bd6334;padding:8px 18px;background:#fff5ef}
li{margin-bottom:12px}@media(max-width:600px){header,main{padding:20px 14px}h1{font-size:25px}}
</style><header><small>ADVOICE / SYSTEM REVIEW</small><h1>系统审查、修复与跨任务试跑</h1>
<p>代码实现、训练独立性、证据执行和实测结果分别核查。历史结果与本轮试跑严格区分。</p></header><main>
<section><h2>审查结论</h2><p>已修复训练折内参照、可靠度、校准温度、证据编号、失效证据引用、病例路由和任务参考类别等问题。
这些修复提高的是方法可信度；是否提高预测性能，必须看重新运行的结果。</p>
<p class="notice">尚未证明独立 ADvoice 超过 SpeechCARE。四任务试跑关闭了实时 Agent，并复用历史预处理与 B1/B2；
不是完整的最新 Agent 评估，也不是重新提取所有原始音频后的结果。</p></section>
<section><h2>本轮实际重训结果</h2><p>同一历史留出受试者集合，更新监督融合与校准。
准确率差值不等于某一机制的因果增益，小样本结果尤其需要保留不确定性。</p><div class="table"><table>
<tr><th>Dataset / task</th><th>Test n</th><th>Historical accuracy</th><th>Current accuracy</th>
<th>Difference</th><th>Current macro AUROC</th><th>Details</th></tr>'''
    body += ''.join(rows) + '''</table></div></section><section><h2>SpeechCARE 对照</h2>
<p>下表重新核验了历史独立 ADvoice 的完整 412 人官方测试集 ID。对照是论文均值，
不是本轮重新训练 SpeechCARE；已有测试集曾参与历史结果检查，不能作为全新确认性验证。</p>
<div class="table"><table><tr><th>Endpoint</th><th>Historical ADvoice</th><th>Paper mean</th><th>Difference</th></tr>'''
    body += comparison + '</table></div></section>' + pilot_section + '''<section><h2>尚不能关闭的设计问题</h2><ol>
<li>Agent 的状态修正尚未重新执行状态预测器。当前修复阻止无效证据推动修正，但不等于从监督先验中移除该证据。</li>
<li>通用训练入口的专家选择仍共享训练交叉验证池，需要独立校准留出集再验证完整选择链。</li>
<li>进展预测与小规模公开视频任务仍缺少稳定判别能力，不能因为代码跑通而宣称有效。</li>
<li>报告质量的自动评分不能替代医生评测；标签指导的状态干预只能解释为理想条件下的敏感性检查。</li>
<li>配置中的潜在混杂因素不能当作实际病例发现。已提供有来源的病例核查接口，但当前没有经过验证的生产者，未知核查状态不会授权 Agent 改写风险。</li>
</ol></section><section><h2>完整评估图</h2><p>保留每个任务的 Layer A、Layer B 与原有详细报告。
以下图表来自实际保存的试跑产物，没有填入预期结果。</p></section>'''
    body += ''.join(details) + '</main></html>'
    output.write_text(body, encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps({"pilot_records": records, "benchmark": benchmark,
                                                      "locked_prepare_test": pilot}, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
