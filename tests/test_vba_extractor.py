"""VBA standard modules are source code, not VB.NET or generic documents."""

from graphify.detect import FileType, classify_file
from graphify.extract import extract


def test_bas_module_procedures_and_case_insensitive_calls(tmp_path):
    source = tmp_path / "Export.bas"
    source.write_text(
        'Attribute VB_Name = "Accounting"\n'
        "Option Explicit\n"
        "Public Sub Start()\n"
        "    Call helper(42)\n"
        "    ' Call Missing()\n"
        '    Debug.Print "don\'t Call Missing()"\n'
        "End Sub\n"
        "Private Function Helper(value As Long) As Long\n"
        "    Helper = value + 1\n"
        "End Function\n",
        encoding="utf-8",
    )
    assert classify_file(source) == FileType.CODE

    result = extract([source], cache_root=tmp_path)
    labels = {node["id"]: node["label"] for node in result["nodes"]}
    assert {"Export.bas", "Accounting", "Start()", "Helper()"} <= set(labels.values())
    calls = {
        (labels[edge["source"]], labels[edge["target"]])
        for edge in result["edges"]
        if edge["relation"] == "calls"
    }
    assert calls == {("Start()", "Helper()")}
    assert all(node.get("metadata", {}).get("language") == "vba" for node in result["nodes"])


def test_bas_bare_calls_and_properties(tmp_path):
    source = tmp_path / "Module1.bas"
    source.write_text(
        "Sub Entry()\n"
        "    target 1\n"
        "End Sub\n"
        "Sub Target(ByVal value As Long)\n"
        "End Sub\n"
        "Public Property Get Name() As String\n"
        '    Name = "a"\n'
        "End Property\n",
        encoding="utf-8",
    )
    result = extract([source], cache_root=tmp_path)
    labels = {node["id"]: node["label"] for node in result["nodes"]}
    assert "Name()" in labels.values()
    assert any(
        edge["relation"] == "calls"
        and labels[edge["source"]] == "Entry()"
        and labels[edge["target"]] == "Target()"
        for edge in result["edges"]
    )
