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


# Exact 0049d1a snapshot: comparison only, never checkpoint resume.
BASELINE_0049D1A = dict(BASELINE_EC0D06A, **{
    'literature_experiment.py': 'ea7c788e784cf1f15db4102381c10d18f5ed7562e59c5b19892028de75e5f3bb',
    'models/literature_resnet.py': 'f23212c7693513f9f754e72e1d2ae0159cdd24108357b6cd2d6b4b59d13edd48',
    'utils/literature_profiling.py': 'c65b48bbee49551081885911c80a974c2a54bed4ac451219f2252d68148ed22e',
    'models/literature_acmix.py': '78ac9a4fde25de98e015e5f77db9c04e2f5c4984391cc8c240cef372d302a6dd',
    'models/conformer_author.py': 'e7f06522b781dd3b45c0f76a303c80b31cdf190a41c49474c285c10cdf2f0a59',
    'models/literature_conformer.py': 'eb3bdff59b2e7086097319a5609f86a3abe6b9ef2dfdfc7cf4a9c3588006f063',
    'utils/literature_training.py': 'bb18858a34666859b5cb975de822380172f902b475fb1119aef57944088d19e7',
    'utils/literature_fusion_compat.py': '96c072439e206f43916eeacca6d082388a5da8b36b7a70225c2a825e76854e24',
})


def is_pre_twostage_source(hashes):
    normalized = dict(hashes)
    if normalized.get('main.py') in (
            'abb8ce2e957faa9d9147092c3cd56de376203682e3a454d02122dfa0c208f478',
            '9822c18c4d8ba3a557c602e9942dfb6a68d67e105ab06f45ced03b3b0937d6b8'):
        normalized['main.py'] = BASELINE_0049D1A['main.py']
    return normalized == BASELINE_0049D1A


# Exact 392a385 snapshot; old architectures unchanged, comparisons only.
BASELINE_392A385 = dict(BASELINE_0049D1A, **{
    'literature_experiment.py': '7fca4be253a6c0a7ef42116bbb77f71f91b11992ca87fdc1883c564f61cdb79f',
    'models/literature_resnet.py': '11838d834df4ad44e08093c523a2ad6c9983b4e47086fbf5f8a992aa0a6b81d1',
    'utils/literature_fusion_compat.py': '669b5b5bb52049853a8593279f9700c25a3de65892d05e1b81f8beea48b4778c',
    'models/literature_twostage.py': '63153af9e16274ca77618f69fe3fc571769ac8729d77d5f03f44760c774aef4e',
})


def is_pre_prebot_source(hashes):
    normalized = dict(hashes)
    if normalized.get('main.py') in (
            'abb8ce2e957faa9d9147092c3cd56de376203682e3a454d02122dfa0c208f478',
            '9822c18c4d8ba3a557c602e9942dfb6a68d67e105ab06f45ced03b3b0937d6b8'):
        normalized['main.py'] = BASELINE_392A385['main.py']
    return normalized == BASELINE_392A385


# Exact 0c160b3 sources, comparison only (never resume).
BASELINE_0C160B3 = dict(BASELINE_392A385, **{
    'literature_experiment.py': '88af3988c2267df387022db0f3b34731052fff85bc2bc7172e983cb8f6b208cb',
    'models/literature_resnet.py': '2e3ebb38073192edea38b03b6daaf86d536757a79a1bb306f77d47cc499da5d4',
    'utils/literature_fusion_compat.py': '57d211bbd286d2222d4cca608ef579fb94005496ba4789da7a880a700aa5ab79',
    'models/literature_prebot.py': '2dfff7cb2cbb026ee03d78de8fe40256ae2f94daa32521a609324fce7031bb78',
})


def is_pre_mixers_source(hashes):
    normalized = dict(hashes)
    if normalized.get('main.py') in (
            'abb8ce2e957faa9d9147092c3cd56de376203682e3a454d02122dfa0c208f478',
            '9822c18c4d8ba3a557c602e9942dfb6a68d67e105ab06f45ced03b3b0937d6b8'):
        normalized['main.py'] = BASELINE_0C160B3['main.py']
    return normalized == BASELINE_0C160B3
