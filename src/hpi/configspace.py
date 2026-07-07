from pathlib import Path


CONFIGSPACE_8HP_TEXT = """name,type,log,lower,upper,default,distribution,item_0,item_1,item_2,item_3,item_4,ordered,distribution_mu,distribution_sigma,distribution_alpha,distribution_beta
lr,float,True,0.02,0.3,0.1,uniform,,,,,,False,,,,
num_emb_type,categorical,,,,pbld,,none,pbld,pl,plr,,False,,,,
add_front_scale,categorical,,,,""" + '"""True"""' + """,,""" + '"""True"""' + """,""" + '"""False"""' + """,,,,False,,,,
p_drop,categorical,,,,""" + '"""0.15"""' + """,,""" + '"""0.0"""' + """,""" + '"""0.15"""' + """,""" + '"""0.3"""' + """,,,False,,,,
wd,categorical,,,,""" + '"""0.02"""' + """,,""" + '"""0.0"""' + """,""" + '"""0.02"""' + """,,,,False,,,,
plr_sigma,float,True,0.05,0.5,0.2,uniform,,,,,,False,,,,
hidden_sizes,categorical,,,,256x3,,256x3,64x5,512x1,,,False,,,,
act,categorical,,,,selu,,selu,mish,relu,,,False,,,,
"""


CONFIGSPACE_9HP_TEXT = CONFIGSPACE_8HP_TEXT + """ls_eps,categorical,,,,""" + '"""0.1"""' + """,,""" + '"""0.0"""' + """,""" + '"""0.1"""' + """,,,,False,,,,
"""


def write_configspace_csv(
    run_dir: Path,
    overwrite: bool = True,
    include_ls_eps: bool = False,
) -> None:
    path = run_dir / "configspace.csv"

    if path.exists() and not overwrite:
        print(f"[SKIP] configspace.csv exists: {path}")
        return

    text = CONFIGSPACE_9HP_TEXT if include_ls_eps else CONFIGSPACE_8HP_TEXT

    path.write_text(text, encoding="utf-8")
    print(f"[CONFIGSPACE] wrote {path}")


def write_configspace_for_all_runs(
    root: Path,
    overwrite: bool = True,
    include_ls_eps: bool = False,
) -> None:
    if not root.exists():
        print(f"[SKIP] Missing root: {root}")
        return

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        trials_csv = run_dir / "trials.csv"
        if not trials_csv.exists():
            continue

        write_configspace_csv(
            run_dir,
            overwrite=overwrite,
            include_ls_eps=include_ls_eps,
        )