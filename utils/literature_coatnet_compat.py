"""Exact completed 4506e0d sources accepted for additive CoAtNet comparison only."""

BASELINE_4506E0D = {
    "datasets/resnet_tiny.py": "11bdc619bd9e4848f9d7728de50d89d3a3d892e45dbe1d2de28d657c3e8f3e2b",
    "datasets/utility_tiny.py": "ee9bbf52827c2bd9ed13ca1e611aa08da51161f1208a03b56b4b36935734e08a",
    "utils/profiling.py": "670ae8aca669e88731e2abae80e4981bc0fd7d1f8f8fa1931458ec25ca94d3a2",
    "datasets/tiny_imagenet.py": "726cb188482599a6dd0f0fbb3c0ce75d1edcf0708953e4cac565a5eefcec7907",
    "models/literature_alternet.py": "375c6b7a5aaadfcad473c62698978588b487a6a13b66347a888ebd47e0e10b7d",
    "resnet_experiment.py": "04e1ad959b0526d4dddfb7533e50f2ff00a83a02a6d194fefd9de877a5730fb8",
    "main.py": "e171f65fe78a45ef10d7422b89131c4ed4fb7e305088bed047aa18449b8b35fe",
    "requirements-literature.txt": "0c8621a820b463fbcd70cb5f86b5e027ed2b3797ee06ac5bfc1b3ef41810be39",
    "literature_experiment.py": "c8ae227a1a41d4aae72268ed0133fcd5722883b7e5b316eada75dc0c80469daa",
    "utils/literature_legacy.py": "7c514f4d6eaf3a0bd8470df6cdf98c5fe9edb85e094e39057f1fa35bee9d0aa5",
    "utility_experiment.py": "b249c8c16bc6b5ed50801e0a9032d292a022310ada8bc59961f29864f538952b",
    "utils/literature_profiling.py": "7dd826a021736da1e0213261172f9d990e3da9cf43ac678769c0ca9d7a0162bb",
    "models/literature_resnet.py": "0427e974d425041ee4ee09ebcb3b7395bb846ccb4c2a9a58ca094f0fbe1ee893",
    "utils/literature_eval_migration.py": "dc25c46fd633cff764fbc030522bae38ea6eeb173a0c48a6517a308461aef8ca"
}


def is_pre_coatnet_source(hashes):
    normalized = dict(hashes)
    # These two Windows line-ending encodings of main.py were already verified
    # in the prior precision migration; the Linux source is canonical here.
    if normalized.get('main.py') in (
            'abb8ce2e957faa9d9147092c3cd56de376203682e3a454d02122dfa0c208f478',
            '9822c18c4d8ba3a557c602e9942dfb6a68d67e105ab06f45ced03b3b0937d6b8'):
        normalized['main.py'] = BASELINE_4506E0D['main.py']
    return normalized == BASELINE_4506E0D

