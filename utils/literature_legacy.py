"""Exact pre-fix sources permitted ONLY for reusing completed CNN controls.

Snapshot: ff14146. LF and CRLF forms are equivalent. This is not a general
ignore-source-check option and never permits resuming old attention training.
"""
LEGACY_SOURCES = {
    "literature_experiment.py": [
        "03ab942dc0ae186fc4f73192fb3676cbc6730982ec7f2bd3910374cb10db8140",
        "61c391f55cbbb4b4e7a7b65373181aa417db852d341af6d78d88e267be391126"
    ],
    "models/literature_resnet.py": [
        "f5bf81429ab367ee3e16a740bb0f614f8c12f463e2ffb2a57fececc6f0589ab1",
        "350f2898f45f70eed606566caf6392821f6e597ff69cc78de4d85f050f845065"
    ],
    "models/literature_alternet.py": [
        "f3a6a3996deefffc5ebf7af55c85cd19597385b967c9d05c893f377f3c48b685",
        "df15f5d21b81a75b07b41a9d5abdcb7d126fc8374305ac5d46b6b1287c331b4c"
    ],
    "utils/literature_profiling.py": [
        "7dd826a021736da1e0213261172f9d990e3da9cf43ac678769c0ca9d7a0162bb",
        "4d736907d87136ae8195ca87c20a7a61022a78ec6787b4eb3f65627fbdc04a5a"
    ],
    "resnet_experiment.py": [
        "04e1ad959b0526d4dddfb7533e50f2ff00a83a02a6d194fefd9de877a5730fb8",
        "88fdbdc89f80049583a6019ff36a64f12f5c1dd993dcc72371b3ec837247de20"
    ],
    "datasets/resnet_tiny.py": [
        "11bdc619bd9e4848f9d7728de50d89d3a3d892e45dbe1d2de28d657c3e8f3e2b",
        "9b5e6da84050b090d38147d0de5fb29534f33b9e3651ea098e918a6a97cd1a48"
    ],
    "datasets/utility_tiny.py": [
        "ee9bbf52827c2bd9ed13ca1e611aa08da51161f1208a03b56b4b36935734e08a",
        "b164e084599185fbf866ecb3ffe44a751a9d70ae30b91faed3587c8ec438b648"
    ],
    "datasets/tiny_imagenet.py": [
        "726cb188482599a6dd0f0fbb3c0ce75d1edcf0708953e4cac565a5eefcec7907",
        "e453f1f71d380201f74978e0bfa592a902285773f3fe07b8b449c8f27d5d76dc"
    ],
    "main.py": [
        "e171f65fe78a45ef10d7422b89131c4ed4fb7e305088bed047aa18449b8b35fe",
        "9822c18c4d8ba3a557c602e9942dfb6a68d67e105ab06f45ced03b3b0937d6b8"
    ],
    "utility_experiment.py": [
        "b249c8c16bc6b5ed50801e0a9032d292a022310ada8bc59961f29864f538952b",
        "27875e667c8fd4e5a1aa2e2ad4cb353f36de5f47a93ac82445aa15bc1e537c4e"
    ],
    "utils/profiling.py": [
        "670ae8aca669e88731e2abae80e4981bc0fd7d1f8f8fa1931458ec25ca94d3a2",
        "b30c4e0446ed54eb1f7b1687f3bd8873df34e8a49ab77e6687471e7e0c9b9713"
    ],
    "requirements-literature.txt": [
        "0c8621a820b463fbcd70cb5f86b5e027ed2b3797ee06ac5bfc1b3ef41810be39",
        "a4c89393fa7ff988d3d73e944f22e8e955766dd974e9f78368d9f7c8fae8a948"
    ]
}


def is_legacy_source(hashes):
    return (set(hashes) == set(LEGACY_SOURCES)
            and all(value in LEGACY_SOURCES[path] for path, value in hashes.items()))
