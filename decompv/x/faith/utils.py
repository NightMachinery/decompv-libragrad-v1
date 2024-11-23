from decompv.x.faith.bootstrap import *
from decompv.x.gbrand import gbrand_from_preset
from pynight.common_regex import float_pattern
from pynight.common_dict import simple_obj
import pandas as pd


faith_default_metrics = [
    "accuracy",
    "accuracy_var_of_sample_mean",
    "recall_mean",
    "recall_var_of_sample_mean",
    "f1_mean",
    "f1_var_of_sample_mean",
    "aopc_mean",
    "aopc_var_of_sample_mean",
    "lodds_mean",
    "lodds_var_of_sample_mean",
]


##
def df_drop(df, columns, ignore_nonexistent_p=True):
    if ignore_nonexistent_p:
        columns = [col for col in columns if col in df.columns]

    return df.drop(columns=columns)


def h_process_df(df):
    mapper = {"area-under-curve": "AUC"}

    if True:
        df = df.rename(columns=mapper)
        df = df_drop(
            df=df,
            columns=[
                "average",
                0,
                1,
                7.5,
                95,
                100,
            ],
        )
        df.columns = df.columns.astype(str)
    else:
        df = df.rename(columns=mapper, level=1)

        drop_columns = [
            (metric, column)
            for metric in df.columns.levels[0]
            for column in ["average", 0]
        ]
        df = df.drop(columns=drop_columns, errors="ignore")

        # Convert the inner column names to strings and reconstruct multi-index
        tuples = [(outer, str(inner)) for outer, inner in df.columns]
        df.columns = pd.MultiIndex.from_tuples(tuples)

    return df


def title_case_index_names(df):
    if df.index.name:
        df.index.name = df.index.name.title()
    if isinstance(df.index, pd.MultiIndex):
        df.index.names = [name.title() if name else name for name in df.index.names]


def df_to_latex_v1(
    df,
    *,
    dest,
    # plabel,
    plabel="Pct. Discarded",
    font_size=8,
    # hspace = "-1.5cm",
    caption=None,
    label=None,
):
    mkdir(dest, do_dirname=True)

    baseline_skip = font_size * 1.2

    df = h_process_df(df)

    df.columns.name = plabel
    title_case_index_names(df)

    def bold_max_by_metric(s, metric="aopc"):
        filtered_series = s[
            s.index.get_level_values("Metric").str.contains(metric, case=False)
        ]

        metric_name = filtered_series.index[0][1]
        if "uparrow" in metric_name:
            m = s == filtered_series.max()
        elif "downarrow" in metric_name:
            m = s == filtered_series.min()
        else:
            raise Exception(f"Unsupported metric name: {metric_name}")

        # ic(s.index)
        return ["font-weight: bold" if v else "" for v in m]

    df_styled = df.style
    for metric in [
        "aopc",
        "lodds",
        "accuracy",
    ]:
        df_styled = df_styled.apply(
            partial(
                bold_max_by_metric,
                metric=metric,
            ),
            axis="index",
        )
    df_styled = df_styled.format(
        {
            column: "{:.2f}"
            for column in df.select_dtypes(include=["float64", "float32"]).columns
        }
    )

    with open(dest, "w") as f:
        f.write("\\begin{table}[h!]\n")

        # if hspace:
        #     f.write(f"\\hspace*{{{hspace}}}\n")

        if font_size:
            f.write(f"\\fontsize{{{font_size}}}{{{baseline_skip}}}\selectfont\n")

        f.write("\\centering\n~\\clap{\n")
        #: [[https://tex.stackexchange.com/questions/600482/how-to-center-the-table-on-the-page-without-following-the-margin][how to center the table on the page without following the margin - TeX - LaTeX Stack Exchange]]

        if True:
            latex_output = df_styled.to_latex(
                convert_css=True,
                clines="skip-last;data",
                hrules=True,
            )
        else:
            latex_output = df.to_latex(
                index=True,
                float_format="{:.2f}".format,
                escape=False,
                multirow=True,
            )

        f.write(
            latex_output,
        )
        f.write("}\n")

        if caption:
            f.write(f"\\caption{{{caption}}}\n")

        if label:
            f.write(f"\\label{{{label}}}\n")

        f.write("\\end{table}\n")

        if font_size:
            f.write("\\normalfont\n")  # This restores the font to the normal size


##
def xp_cls_metrics_to_table_v2(
    df,
    metrics=["accuracy"],
    sort_opts="auto",
    high_accu_p=False,
):
    up_arrow = r"\(\;\uparrow\)"
    down_arrow = r"\(\;\downarrow\)"
    if high_accu_p:
        accu_arrow = up_arrow
        aopc_arrow = down_arrow
    else:
        accu_arrow = down_arrow
        aopc_arrow = up_arrow

    metric_names = {
        "aopc_mean": f"AOPC {aopc_arrow}",
        "lodds_mean": f"LOdds {aopc_arrow}",
        "accuracy": f"Accuracy {accu_arrow}",
    }

    # Define a helper function to calculate statistics for each metric
    def calc_stats(pivot_df, x_values):
        pivot_df["area-under-curve"] = pivot_df.apply(
            lambda row: np.trapz(row, x=x_values), axis=1
        )
        pivot_df["average"] = pivot_df.drop(columns=["area-under-curve"]).mean(axis=1)
        return pivot_df

    result_dfs = []

    # Iterate over metrics and calculate statistics
    for metric in metrics:
        # Filter dataframe by the current metric
        df_metric = df[df["metric"] == metric]

        # Pivot the table to get methods as rows and top-ratio as columns
        pivot_df = df_metric.pivot(index="method", columns="top-ratio", values="value")

        # Convert the column names to numeric
        x_values = pivot_df.columns.values.astype(float) / 100

        # Calculate statistics for the current metric
        pivot_df = calc_stats(pivot_df, x_values)

        # Set multi-index with the metric name for the rows
        pivot_df.index = pd.MultiIndex.from_product(
            [
                pivot_df.index,
                [metric_names[metric]],
            ],
            names=[
                "method",
                "metric",
            ],
        )

        result_dfs.append(pivot_df)

    # Join all the metric-specific dataframes
    final_df = pd.concat(result_dfs)
    # final_df = final_df.groupby(['Method'])

    if sort_opts == "auto":
        sort_opts = dict()

    if sort_opts is not None:
        final_df = df_sort(final_df, **sort_opts)

    return final_df


##
def register_method(
    *,
    name=None,
    name_fn=None,
    official_name,
    primary_key=None,
    gradient_brand=None,
    gbrand_preset=None,
    methods_dict,
    model_names,
    **kwargs,
):
    if primary_key is None:
        primary_key = official_name

    if primary_key not in methods_dict:
        methods_dict[primary_key] = defaultdict_defaultdict()

    method_obj = dict(
        official_name=official_name,
        primary_key=primary_key,
        qual=defaultdict_defaultdict(),
        **kwargs,
    )
    if name is not None:
        method_obj["name"] = name

    elif name_fn is not None:
        method_obj["name_fn"] = name_fn

    else:
        raise Exception("Either 'name' or 'name_fn' must be provided.")

    if gradient_brand is None:
        assert (
            gbrand_preset
        ), "Eithr 'gradient_brand' or 'gbrand_preset' must be provided."

        method_obj["gbrand_fn"] = partial(
            gbrand_from_preset,
            gbrand_preset=gbrand_preset,
        )

    else:
        method_obj["gradient_brand"] = gradient_brand

    methods_dict[primary_key].update(method_obj)

    for model_name_ in model_names:
        methods_dict[primary_key].setdefault(model_name_, defaultdict(dict))

    return methods_dict


def get_blocks_len(model_name):
    #: [[id:99f77534-bb9c-4bec-82c3-849b690df3d3][@table @ViT @architecture @sizes Vision Transformer Model Variants]]
    ##
    if (
        any(
            pat in model_name
            for pat in [
                "vit_so400m",
            ]
        )
        or model_name.startswith("ViT-SO400M")
        or model_name
        in [
            "vit_so400m_patch14_siglip_378.webli_ft_in1k",
            "vit_so400m_patch14_siglip_gap_378.webli_ft_in1k",
            #: [[file:~/code/uni/pytorch-image-models/timm/models/vision_transformer.py::def vit_so400m_patch14_siglip_gap_384(pretrained: bool = False, **kwargs) -> VisionTransformer:]]
        ]
    ):
        return 27

    elif any(
        pat in model_name
        for pat in [
            "small",
            "tiny",
            "base",
        ]
    ):
        return 12

    elif any(
        pat in model_name
        for pat in [
            "large",
            "EVA02-L-",
        ]
    ) or model_name in [
        "gmixer_24_224.ra3_in1k",
    ]:
        return 24

    elif any(
        pat in model_name
        for pat in [
            "huge",
        ]
    ):
        return 32

    else:
        raise Exception(f"Unknown model name: {model_name}")


def get_gbrand(*args, **kwargs):
    gbrand = h_get_gbrand(*args, **kwargs)
    return gbrand


def h_get_gbrand(
    *,
    method_obj,
    model_name,
    gbrand_postfix=None,
):
    if "gradient_brand" in method_obj:
        assert "gbrand_fn" not in method_obj

        return method_obj["gradient_brand"]

    elif "gbrand_fn" in method_obj:
        assert "gradient_brand" not in method_obj

        name = method_obj["gbrand_fn"](
            method_obj=method_obj,
            model_name=model_name,
            gbrand_postfix=gbrand_postfix,
        )

        return name

    else:
        raise Exception(
            f"Method object does not have 'gradient_brand' or 'gbrand_fn':\n{method_obj}"
        )


def get_method_name(method_obj, model_name):
    if "name" in method_obj:
        assert "name_fn" not in method_obj

        return method_obj["name"]

    elif "name_fn" in method_obj:
        assert "name" not in method_obj

        blocks_len = get_blocks_len(model_name)

        name = method_obj["name_fn"](
            method_obj, model_name=model_name, blocks_len=blocks_len
        )

        # if method_obj["primary_key"] == "DecompX+ Half":
        #     ic(name, blocks_len)

        return name

    else:
        raise Exception(
            f"Method object does not have 'name' or 'name_fn':\n{method_obj}"
        )


def name_fn_factory(name_template):
    def name_fn(
        method_obj,
        *,
        model_name,
        blocks_len,
        **kwargs,
    ):
        name = name_template
        name = name.replace("LAST_BLOCK", str(blocks_len - 1))
        name = name.replace("HALF_BLOCK", str(blocks_len // 2))
        return name

    return name_fn


##
def gradient_brand_official_name_get(x):
    return x


##
def plt_export(
    fig,
    formats=[
        # "png",
        "pdf",
    ],
    show_p=True,
    close_p=True,
    tlg_p=False,
    clipboard_p=False,
    path=None,
    tight_layout_p=False,
    dpi=150,
):
    if tight_layout_p:
        fig.tight_layout()

    saved_paths = []
    for format in formats:
        if path is None:
            dest = tempfile.mktemp(suffix=f".{format}")
        else:
            dest = f"{path}.{format}"
            mkdir(dest, do_dirname=True)

        fig.savefig(
            dest,
            format=format,
            dpi=dpi,
        )
        saved_paths.append(dest)

        if clipboard_p:
            z("pa {dest}")

        if tlg_p:
            # ic(f"tsendf Arstar {dest}")
            # ic(z("pxa tsendf Arstar {dest}"))
            z("awaysh tsendf Arstar -- {dest}")

    if show_p:
        # fig.show()
        print("")
        plt.show()

    if close_p:
        plt.close()

    return saved_paths


##
def faith_raw_dict_load(
    *,
    metrics_model_name,
    end_pat,
):
    metrics_model_name_escaped = metrics_model_name.replace(
        "_",
        r"\_",
    )

    selected = list_children(
        ic(f"{CLS_METRICS_ROOT}/{metrics_model_name}/"),
        # f"{CLS_METRICS_ROOT}/vit_base_patch16_224.augreg2_in21k_ft_in1k/",
        # f"{METRICS_ROOT}/cls_tmp_01",
        abs_include_patterns=[f"{end_pat}"],
        # abs_include_patterns=[f'ALTI.*{end_pat}'],
        # abs_include_patterns=[f'DecompV.*{end_pat}'],
        # abs_include_patterns=[f'globenc.*{end_pat}'],
        # abs_include_patterns=[f'attributions_s_(?:blocks__(?:\\d+)__)?CAT(?!_AttnFrom).*{end_pat}'],
        # abs_include_patterns=[f'attributions_s_(?:blocks__(?:\\d+)__)?CAT_AttnFrom.*{end_pat}'],
        # abs_include_patterns=[f'attributions_s_(?:blocks__(?:\\d+)__)?MeanReLU__AttnGrad_Attn.*{end_pat}'],
        # abs_include_patterns=[f'_logit.*{end_pat}'],
        # abs_include_patterns=[f'(_nratio).*{end_pat}'],
        # abs_include_patterns=[f'_ro.*{end_pat}'],
        # abs_include_patterns=[f'(_logit|IxG|IG).*{end_pat}'],
        # abs_include_patterns=[f'.*CAT_AttnFrom.*_sum.*{end_pat}'],
        # abs_include_patterns=[f'(?:attributions_s_CAT_sum|attributions_s_CAT_AttnFrom_sum|attributions_s_MeanReLUAttnGrad_MeanAttn_relu_ro_str50|attributions_s_blocks__10__MeanReLUAttnGrad){end_pat}'],
        # |attributions_s_MeanReLUAttnGrad_MeanAttn_CAT_relu_to1_ro_str50
        recursive=True,
    )

    ic(len(selected))
    if len(selected) == 0:
        return None

    xp_cls_metrics = json_partitioned_load(selected)

    return xp_cls_metrics


def faith_raw_dict_process(
    xp_cls_metrics,
    scale_by=100,
):
    xp_cls_topratio_metrics_df = metric_dict_to_df_topratio(
        xp_cls_metrics,
        officialize_names=False,
        scale_by=scale_by,
    )
    #: `xp_cls_topratio_metrics_df.head(2)`
    #:    top-ratio                         method     metric     value
    #: 0      100.0  CAT_s:L2_AttnFrom_sum_f6_to11        len  100000.0
    #: 1      100.0  CAT_s:L2_AttnFrom_sum_f6_to11  aopc_mean       0.0

    xp_cls_nratio_metrics_df = metric_dict_to_df_nratio(
        xp_cls_metrics,
        officialize_names=False,
        scale_by=scale_by,
    )

    print("Processed the raw faith metrics into proper dataframes.")

    return xp_cls_topratio_metrics_df, xp_cls_nratio_metrics_df


def faith2tables(
    *,
    name,
    metrics_model_name,
    xp_cls_topratio_metrics_df,
    xp_cls_nratio_metrics_df,
    save_p=True,
    metrics=faith_default_metrics,
):
    tables = dict()
    for metric in metrics:
        tables[f"xp_cls_topratio_table_{metric}"] = xp_cls_metrics_to_table(
            xp_cls_topratio_metrics_df,
            metric=metric,
        )

        if save_p:
            df_to_html2(
                tables[f"xp_cls_topratio_table_{metric}"],
                f"/opt/static/{metrics_model_name}/topratio/{metric}/{name}.html",
            )
            tables[f"xp_cls_topratio_table_{metric}"].to_csv(
                f"/opt/static/{metrics_model_name}/topratio/{metric}/{name}.csv",
                index=True,
            )

            tables[f"xp_cls_nratio_table_{metric}"] = xp_cls_metrics_to_table(
                xp_cls_nratio_metrics_df,
                metric=metric,
            )

            df_to_html2(
                tables[f"xp_cls_nratio_table_{metric}"],
                f"/opt/static/{metrics_model_name}/nratio/{metric}/{name}.html",
            )
            tables[f"xp_cls_nratio_table_{metric}"].to_csv(
                f"/opt/static/{metrics_model_name}/nratio/{metric}/{name}.csv",
                index=True,
            )
            # df_to_latex_v1(
            #     tables[f"xp_cls_nratio_table_{metric}"],
            #     dest=f"/opt/static/{metrics_model_name}/nratio/{metric}/{name}.tex",
            #     label=f"tab:{name}_nratio",
            #     caption=f"Percentage of Top Patches Discarded for Different Layers of {name}",
            # )

    return tables


def metric_name_to_var_of_sample_mean(metric):
    if metric.endswith("_mean"):
        metric_var_of_sample_mean = f"""{metric[:-len("_mean")]}_var_of_sample_mean"""

    else:
        metric_var_of_sample_mean = f"{metric}_var_of_sample_mean"

    return metric_var_of_sample_mean


def faith_table_to_dict(
    *,
    tables,
    metric,
    group,
):
    table = tables[f"xp_cls_{group}_table_{metric}"]

    metric_var_of_sample_mean = metric_name_to_var_of_sample_mean(metric)
    table_var_of_sample_mean = tables[
        f"xp_cls_{group}_table_{metric_var_of_sample_mean}"
    ]

    col_name_mapping = {
        "area-under-curve": "auc",
        "average": "average",
    }

    result = {}

    for index, row in table.iterrows():
        method = index
        method_data = {}

        row_var_of_sample_mean = table_var_of_sample_mean.loc[index]
        for col, value in row.items():
            col_var_of_sample_mean = row_var_of_sample_mean[col]

            if (re.match(float_pattern, str(col))) and int(col) == col:
                col = int(col)

            if col in col_name_mapping:
                col = col_name_mapping[col]

            method_data[col] = value
            method_data[f"{col}_var_of_sample_mean"] = col_var_of_sample_mean

        result[method] = method_data

    return result


def faith_process_pipeline(
    *,
    name,
    metrics_model_name,
    end_pat,
    metrics=faith_default_metrics,
    save_p=True,
    **kwargs,
):
    xp_cls_metrics = faith_raw_dict_load(
        metrics_model_name=metrics_model_name,
        end_pat=end_pat,
    )
    if xp_cls_metrics is None:
        return None

    xp_cls_topratio_metrics_df, xp_cls_nratio_metrics_df = faith_raw_dict_process(
        xp_cls_metrics
    )

    tables = faith2tables(
        name=name,
        metrics_model_name=metrics_model_name,
        xp_cls_topratio_metrics_df=xp_cls_topratio_metrics_df,
        xp_cls_nratio_metrics_df=xp_cls_nratio_metrics_df,
        metrics=metrics,
        save_p=save_p,
        **kwargs,
    )

    dicts = dict()
    for metric in metrics:
        if metric.endswith("_var_of_sample_mean"):
            continue

        for group in ["topratio", "nratio"]:
            curr_faith_dict = faith_table_to_dict(
                tables=tables,
                metric=metric,
                group=group,
            )
            curr_dict_name = f"{group}_{metric}"
            dicts[curr_dict_name] = curr_faith_dict

            if save_p:
                dest = f"/opt/static/{metrics_model_name}/{group}/{metric}/{name}.json"
                print(f"\n{curr_dict_name}: saving to:\n  {dest}")

                json_save(
                    curr_faith_dict,
                    file=dest,
                    exists="ignore",
                )

    return simple_obj(
        xp_cls_metrics=xp_cls_metrics,
        xp_cls_topratio_metrics_df=xp_cls_topratio_metrics_df,
        xp_cls_nratio_metrics_df=xp_cls_nratio_metrics_df,
        tables=tables,
        dicts=dicts,
    )


##
