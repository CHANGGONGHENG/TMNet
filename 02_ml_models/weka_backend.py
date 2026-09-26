"""WEKA backend for the classifiers that have no scikit-learn equivalent.

RotationForest and LogitBoost are provided by the WEKA toolkit (Java). This module runs them
through ``python-weka-wrapper3``: the feature tables are written to ARFF, the classifier is trained
in the JVM and the predicted probabilities are returned to Python.

Requirements
------------
* a Java runtime (JDK or JRE, version 8 or newer), located through ``JAVA_HOME`` or ``--java-home``
* ``pip install python-weka-wrapper3`` (which bundles ``weka.jar``)
* for RotationForest: the WEKA package ``rotationForest`` installed in ``WEKA_HOME``, e.g.
  ``python -c "import weka.core.jvm as j; from weka.core import packages; j.start(); packages.install_package('rotationForest'); j.stop()"``

Environment variables
---------------------
``JAVA_HOME``   path to the Java runtime
``WEKA_HOME``   directory holding the installed WEKA packages (default: ``<repo>/.wekafiles``)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

JVM_STARTED = False

# WEKA classifiers used by ml_config.yaml (backend: weka)
WEKA_CLASSIFIERS = {
    "RotationForest": "weka.classifiers.meta.RotationForest",
    "LogitBoost": "weka.classifiers.meta.LogitBoost",
    "ConditionalInferenceTree": "weka.classifiers.trees.ConditionalInferenceTree",
    "GentleBoost": "weka.classifiers.meta.GentleBoost",
}


def _weka_home() -> str:
    here = Path(__file__).resolve().parent
    default = here.parent / ".wekafiles"
    home = os.environ.get("WEKA_HOME") or str(default)
    os.makedirs(home, exist_ok=True)
    return home


def is_available() -> bool:
    """True when the wrapper and a Java runtime are available."""
    try:
        import weka.core.jvm  # noqa: F401
    except ImportError:
        return False
    java_home = os.environ.get("JAVA_HOME", "")
    if java_home and Path(java_home, "bin").exists():
        return True
    return bool(os.environ.get("PATH"))


def start_jvm(java_home: str | None = None, weka_home: str | None = None) -> None:
    """Start the JVM once, with the WEKA packages on the classpath."""
    global JVM_STARTED
    if JVM_STARTED:
        return
    if java_home:
        os.environ["JAVA_HOME"] = java_home
    home = weka_home or _weka_home()
    os.environ["WEKA_HOME"] = home

    import glob

    import weka.core.jvm as jvm

    package_jars = glob.glob(os.path.join(home, "packages", "**", "*.jar"), recursive=True)
    jvm.start(packages=home, class_path=package_jars, logging_level=40)
    JVM_STARTED = True


def stop_jvm() -> None:
    global JVM_STARTED
    if not JVM_STARTED:
        return
    import weka.core.jvm as jvm

    jvm.stop()
    JVM_STARTED = False


def _write_arff(path: Path, x: np.ndarray, y: np.ndarray | None, feature_names: list[str]) -> None:
    """Write a dense numeric ARFF file; the class attribute is last and nominal."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("@relation tmnet\n\n")
        for name in feature_names:
            fh.write(f"@attribute {_safe(name)} numeric\n")
        if y is not None:
            fh.write("@attribute pCR {0,1}\n")
        fh.write("\n@data\n")
        for i in range(x.shape[0]):
            values = ",".join("?" if not np.isfinite(v) else repr(float(v)) for v in x[i])
            if y is not None:
                values += f",{int(y[i])}"
            fh.write(values + "\n")


def _safe(name: str) -> str:
    """WEKA attribute names must not contain spaces or the ARFF special characters."""
    for ch in " \t{}%,'\"\\":
        name = name.replace(ch, "_")
    return name or "feature"


def fit_predict(
    classname: str,
    options: list[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict_list: list[np.ndarray],
    feature_names: list[str],
    workdir: Path,
) -> list[np.ndarray]:
    """Train a WEKA classifier and return P(class = 1) for each prediction matrix."""
    from weka.classifiers import Classifier
    from weka.core.converters import Loader

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    train_arff = workdir / "train.arff"
    _write_arff(train_arff, x_train, y_train, feature_names)

    loader = Loader(classname="weka.core.converters.ArffLoader")
    train_data = loader.load_file(str(train_arff))
    train_data.class_is_last()

    model = Classifier(classname=classname, options=options)
    model.build_classifier(train_data)

    results = []
    for i, x in enumerate(x_predict_list):
        arff = workdir / f"predict_{i}.arff"
        # the class attribute is written with a placeholder value so that every file shares the
        # same header as the training file and can be scored directly by WEKA
        _write_arff(arff, x, np.zeros(x.shape[0], dtype=int), feature_names)
        data = loader.load_file(str(arff))
        data.class_is_last()
        probs = np.zeros(x.shape[0], dtype=float)
        for j in range(data.num_instances):
            inst = data.get_instance(j)
            dist = model.distribution_for_instance(inst)
            probs[j] = float(dist[1])
        results.append(probs)
    return results
