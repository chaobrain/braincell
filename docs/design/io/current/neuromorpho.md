# NeuroMorpho API

`braincell.io` 提供按 neuron ID 加载、检索、下载和缓存 NeuroMorpho 数据的接口。
加载返回 Morphology，fetch/download 返回文件记录。SWC 导入模式由 [IO API](api.md) 解释。

## 离线可运行示例

以下注入仓库测试中的 HTTP 替身，展示查询结果和下载计划的读取；实际使用时省略 session 即可访问服务。

```python
from tempfile import TemporaryDirectory
import braincell as bc
from braincell.io.neuromorpho._testing import FakeSession, FakeResponse, sample_neuron_payload

payload = {
    "_embedded": {"neuronResources": [sample_neuron_payload()]},
    "page": {"number": 0, "size": 20, "totalPages": 1, "totalElements": 1},
}
session = FakeSession([FakeResponse(json_data=payload)])
with TemporaryDirectory() as directory:
    client = bc.io.NeuroMorphoClient(session=session, cache_dir=directory)
    page = client.search("species:mouse")
    neuron = page.items[0]
    plans = client.file_plan(neuron, mode="standard")
    assert neuron.neuron_id == 10047
    assert len(plans) == 1
    assert plans[0].filename.endswith(".swc")
    assert len(session.calls) == 1
```

## 一步加载与下载

```text
load_neuromorpho(neuron_id, *, cache_dir=None, mode="neuromorpho", client=None,
                 return_report=False, overwrite=False) -> Morphology | (Morphology, SwcReport)
Morphology.from_neuromorpho(neuron_id, *, cache_dir=None, mode="neuromorpho", client=None,
                          return_report=False, overwrite=False)
fetch_neuromorpho(neuron_id, dest=None, *, mode="standard", overwrite=False,
                  client=None) -> NeuroMorphoDownloadRecord
```

neuron_id 是整数服务 ID；client=None 创建默认客户端。load 的 cache_dir=None 使用默认用户缓存目录，
mode 是 **SWC 导入模式**。fetch 的 dest 是输出目录，mode 则是 **文件种类** standard/original/both。
standard 为标准 SWC，original 为归档原文件；原始文件可能是 ASC，fetch 本身不解析它。
overwrite=False 复用已有文件，True 重新下载。return_report=True 让 load 同时返回 SWC 诊断。
load_neuromorpho 还从 braincell 顶层导出。

## Client

```text
NeuroMorphoClient(session=None, *, timeout=30.0, cache_dir=None, retries=3, backoff_base=0.5)
search(query, *, fq=None, size=20, page=0, sort="neuron_id,asc") -> NeuroMorphoSearchPage
iter_search(query, *, fq=None, size=20, limit=None, start_page=0,
            sort="neuron_id,asc") -> Iterator[NeuroMorphoNeuron]
get_neuron(neuron_id) -> NeuroMorphoNeuron
get_measurement(neuron) -> NeuroMorphoMeasurement
get_urls(neuron) -> NeuroMorphoUrls
get_cache_status(neuron) -> NeuroMorphoCacheStatus
describe(neuron, *, include_measurement=True) -> NeuroMorphoDetail
file_plan(neuron, *, mode="both") -> tuple[NeuroMorphoFilePlan, ...]
download(neuron, output_dir=None, *, mode="both", overwrite=False,
         dry_run=False) -> NeuroMorphoDownloadRecord
```

| 参数 | 含义 |
| --- | --- |
| session | requests 风格会话；None 建立默认会话，可注入测试替身 |
| timeout、retries、backoff_base | 超时秒数、重试次数和退避基数；无 brainunit 单位 |
| query、fq | 查询字符串或 NeuroMorphoQuery；fq 为额外过滤字符串列表 |
| size、page、start_page、limit | 每页数量、从 0 开始页号、迭代起点和可选总条数上限 |
| sort | 服务排序字符串；默认按 neuron_id 升序 |
| neuron | get_urls/file_plan 需要 NeuroMorphoNeuron；其余同名参数也接受整数 ID |
| output_dir | 下载目录；None 使用配置的缓存布局 |
| dry_run | 返回下载计划记录而不写目标文件；解析整数 ID 等元数据步骤仍可能访问网络 |

search 返回一页，不隐式遍历；iter_search 按需请求后续页。get_cache_status 读取缓存状态，
describe 将 neuron、measurement、urls、cache_status 汇成一个对象。

## 结果与缓存

| 类型 | 主要字段 |
| --- | --- |
| NeuroMorphoNeuron | neuron_id、neuron_name、archive、species、brain_region、cell_type、原 payload |
| NeuroMorphoSearchPage | items、page、size、total_pages、total_elements、query_url |
| NeuroMorphoFilePlan | kind、url、filename、skip、reason |
| NeuroMorphoDownloadRecord | folder、metadata_path、download_items、measurement、download_mode、dry_run |
| NeuroMorphoDownloadItem | path、downloaded_now、kind、url、reason |
| NeuroMorphoCacheStatus | configured、folder、exists、metadata_exists、standard_exists、original_exists |
| NeuroMorphoMeasurement | length、surface、volume、path_distance 等服务数值，以及 raw/extras |

Measurement 保留服务的数值字段，不能直接当作 brainunit Quantity；与本地几何比较时需按字段单位转换。
独立缓存接口从 braincell.io 导入：

```text
NeuroMorphoCache(root)
cache.list_neurons() -> tuple[int, ...]
cache.contains(neuron_id) -> bool
cache.status(neuron_id, *, neuron_name=None, original_format=None) -> NeuroMorphoCacheStatus
cache.metadata(neuron_id) -> Mapping
cache.measurement(neuron_id) -> NeuroMorphoMeasurement | None
cache.standard_swc_path(neuron_id) -> Path | None
cache.original_file_path(neuron_id) -> Path | None
cache.load(neuron_id, *, mode="neuromorpho", return_report=False)
cache.remove(neuron_id) -> bool
cache.clear() -> int
```

root 是本地缓存根目录。load 只解析已缓存的标准 SWC，缺少文件时失败，不隐式下载；
remove 删除一个 neuron 目录并返回是否存在，clear 删除缓存记录并返回数量。
这些删除操作立即修改本地缓存。路径布局和元数据文件见
[cache.py](../../../../braincell/io/neuromorpho/cache.py)。

```text
NeuroMorphoQuery(species=None, brain_region=None, cell_type=None, archive=None,
                 original_format=None, stain=None, age_classification=None,
                 gender=None, raw_q=(), raw_fq=())
query.to_q() -> str
query.to_fq() -> list[str]
query.to_params() -> dict
```

语义字段接受字符串或字符串元组；同一字段多值用 OR，不同字段用 AND。
raw_q/raw_fq 为原始查询子句元组。转换返回新的查询字符串/参数对象，
不发送请求；实现见 [query.py](../../../../braincell/io/neuromorpho/query.py)。
服务和下载失败使用 NeuroMorphoError 家族；HTTP 错误为 NeuroMorphoHTTPError，404 为 NeuroMorphoNotFoundError。
服务可用性和数据变化与本地 reader 验证是两个环节，离线测试入口为
[client_test.py](../../../../braincell/io/neuromorpho/client_test.py)。完整交互教程见
[neuromorpho.ipynb](../../../../examples/io/neuromorpho.ipynb)。
