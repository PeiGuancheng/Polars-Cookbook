import gc
import time
from statistics import median

import numpy as np
import pandas as pd
import polars as pl

N = 10_000_000
N_GROUPS = 10_000
REPEATS = 5
rng = np.random.default_rng(42)

# 两边使用相同数据，不包含空值，避免空值语义影响比较。
data = {
    "key": rng.integers(0, N_GROUPS, size=N, dtype=np.int64),
    "segment": rng.integers(0, 10, size=N, dtype=np.int64),
    "quantity": rng.integers(1, 100, size=N, dtype=np.int64),
    "amount": rng.random(N) * 1_000,
}

df_pd = pd.DataFrame(data)
df_pl = pl.from_pandas(df_pd)

# 右表键唯一，连接结果不会因为重复键而膨胀。
dim_pd = pd.DataFrame({
    "key": np.arange(N_GROUPS, dtype=np.int64),
    "rating": rng.integers(0, 5, size=N_GROUPS, dtype=np.int64),
})
dim_pl = pl.from_pandas(dim_pd)

# 预先拆分，计时仅包含拼接操作。
parts_pd = [
    df_pd.iloc[start:start + N // 4]
    for start in range(0, N, N // 4)
]
parts_pl = [pl.from_pandas(part) for part in parts_pd]


def pd_window():
    return df_pd.assign(
        group_mean=df_pd.groupby("key")["amount"].transform("mean")
    )


def pl_window():
    return df_pl.with_columns(
        pl.col("amount").mean().over("key").alias("group_mean")
    )


# 统一输出列名，方便检查两边结果是否一致。
cases = {
    "单列分组求和": (
        lambda: df_pd.groupby("key", as_index=False, sort=False).agg(
            total_amount=("amount", "sum")
        ),
        lambda: df_pl.group_by("key").agg(
            pl.col("amount").sum().alias("total_amount")
        ),
        ["key"],
    ),
    "多键多指标聚合": (
        lambda: df_pd.groupby(
            ["key", "segment"], as_index=False, sort=False
        ).agg(
            total_quantity=("quantity", "sum"),
            mean_amount=("amount", "mean"),
            max_amount=("amount", "max"),
            rows=("amount", "size"),
        ),
        lambda: df_pl.group_by("key", "segment").agg(
            pl.col("quantity").sum().alias("total_quantity"),
            pl.col("amount").mean().alias("mean_amount"),
            pl.col("amount").max().alias("max_amount"),
            pl.len().alias("rows"),
        ),
        ["key", "segment"],
    ),
    "分组去重计数": (
        lambda: df_pd.groupby("key", as_index=False, sort=False).agg(
            unique_quantity=("quantity", "nunique")
        ),
        lambda: df_pl.group_by("key").agg(
            pl.col("quantity").n_unique().alias("unique_quantity")
        ),
        ["key"],
    ),
    "分组均值填回原表": (
        pd_window,
        pl_window,
        None,
    ),
    "左连接": (
        lambda: df_pd.merge(
            dim_pd, on="key", how="left", sort=False
        ),
        lambda: df_pl.join(
            dim_pl, on="key", how="left", maintain_order="left"
        ),
        None,
    ),
    "内连接": (
        lambda: df_pd.merge(
            dim_pd, on="key", how="inner", sort=False
        ),
        lambda: df_pl.join(dim_pl, on="key", how="inner"),
        list(df_pd.columns),
    ),
    "纵向拼接": (
        lambda: pd.concat(parts_pd, ignore_index=True),
        lambda: pl.concat(parts_pl),
        None,
    ),
}


def benchmark(func, repeats=REPEATS):
    # 预热，并在计时区间外回收上一轮结果。
    result = func()
    del result

    elapsed = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        result = func()
        elapsed.append(time.perf_counter() - started)
        del result

    return median(elapsed)


def normalize(frame, sort_cols):
    if sort_cols:
        frame = frame.sort_values(sort_cols)
    return frame.reset_index(drop=True)


records = []

for name, (pd_func, pl_func, sort_cols) in cases.items():
    # 正确性校验不纳入计时，允许浮点聚合中的微小舍入差异。
    expected = normalize(pd_func(), sort_cols)
    actual = normalize(pl_func().to_pandas(), sort_cols)

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )
    del expected, actual

    pd_seconds = benchmark(pd_func)
    pl_seconds = benchmark(pl_func)

    records.append({
        "场景": name,
        "Pandas 秒": pd_seconds,
        "Polars 秒": pl_seconds,
        "Pandas / Polars": pd_seconds / pl_seconds,
    })

print(
    f"行数：{N:,}，分组数：{N_GROUPS:,}，重复次数：{REPEATS}\n"
    f"Pandas：{pd.__version__}，Polars：{pl.__version__}\n"
    f"Polars 线程池大小：{pl.thread_pool_size()}"
)

results = pd.DataFrame(records)
print(results.round(4))