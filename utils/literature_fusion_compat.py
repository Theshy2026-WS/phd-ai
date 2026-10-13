"""Exact ec0d06a sources for completed-baseline comparisons, never resume."""

BASELINE_EC0D06A = {
    "literature_experiment.py": "c2dcbc22e9f7827be4f4e8929242465cfca122515334a10aa3ec333d3b897396",
    "models/literature_resnet.py": "5431b5d64e34b3929e1373e97169a27dd2efc25155af5e1825a7b49bb6efb945",
    "models/literature_alternet.py": "375c6b7a5aaadfcad473c62698978588b487a6a13b66347a888ebd47e0e10b7d",
    "utils/literature_profiling.py": "fdf2a7ab7c6f4469958ee5c053ff21a7d0a99e4863f6d361d32c55015f026bb1",
    "resnet_experiment.py": "04e1ad959b0526d4dddfb7533e50f2ff00a83a02a6d194fefd9de877a5730fb8",
    "datasets/resnet_tiny.py": "11bdc619bd9e4848f9d7728de50d89d3a3d892e45dbe1d2de28d657c3e8f3e2b",
    "datasets/utility_tiny.py": "ee9bbf52827c2bd9ed13ca1e611aa08da51161f1208a03b56b4b36935734e08a",
    "datasets/tiny_imagenet.py": "726cb188482599a6dd0f0fbb3c0ce75d1edcf0708953e4cac565a5eefcec7907",
    "main.py": "e171f65fe78a45ef10d7422b89131c4ed4fb7e305088bed047aa18449b8b35fe",
    "utility_experiment.py": "b249c8c16bc6b5ed50801e0a9032d292a022310ada8bc59961f29864f538952b",
    "utils/profiling.py": "670ae8aca669e88731e2abae80e4981bc0fd7d1f8f8fa1931458ec25ca94d3a2",
    "requirements-literature.txt": "0c8621a820b463fbcd70cb5f86b5e027ed2b3797ee06ac5bfc1b3ef41810be39",
    "utils/literature_legacy.py": "7c514f4d6eaf3a0bd8470df6cdf98c5fe9edb85e094e39057f1fa35bee9d0aa5",
    "utils/literature_eval_migration.py": "dc25c46fd633cff764fbc030522bae38ea6eeb173a0c48a6517a308461aef8ca",
    "models/literature_coatnet.py": "a69b9b871580462489baceb8652252c6dd0320b6b26b1db57a3e628347eda7d5",
    "utils/literature_coatnet_compat.py": "06693d272afc735edcdee8ac7be8e15bbc8d33bc0a5ae79d8365b188b7a92534"
}


def is_pre_fusion_source(hashes):
    normalized = dict(hashes)
    if normalized.get("main.py") in (
            "abb8ce2e957faa9d9147092c3cd56de376203682e3a454d02122dfa0c208f478",
            "9822c18c4d8ba3a557c602e9942dfb6a68d67e105ab06f45ced03b3b0937d6b8"):
        normalized["main.py"] = BASELINE_EC0D06A["main.py"]
    return normalized == BASELINE_EC0D06A
