from types import SimpleNamespace

import squadds.simulations.utils_ansys as utils_ansys


def test_setMaterialProperties_prefers_live_renderer_session():
    calls = []

    class FakeDefinitionManager:
        def EditMaterial(self, material_name, args):
            calls.append(("edit", material_name, args))

    class FakeProject:
        def GetDefinitionManager(self):
            calls.append(("definition_manager",))
            return FakeDefinitionManager()

    renderer = SimpleNamespace(
        pinfo=SimpleNamespace(
            project_name="Project1",
            design_name="CavitySweep_hfss",
            desktop=SimpleNamespace(_desktop=object()),
            project=SimpleNamespace(_project=FakeProject(), parent=SimpleNamespace(_desktop=object())),
            design=SimpleNamespace(_design=object()),
        )
    )

    utils_ansys.setMaterialProperties(
        renderer=renderer,
        hfss_factory=lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )

    assert calls[0] == ("definition_manager",)
    assert calls[1][0] == "edit"
    assert calls[1][1] == "silicon"
    assert "permittivity:=" in calls[1][2]
    assert "11.45" in calls[1][2]
    assert "dielectric_loss_tangent:=" in calls[1][2]
    assert "1e-07" in calls[1][2]


def test_get_freq_Q_kappa_passes_live_renderer_to_material_patch(monkeypatch):
    calls = {}

    def fake_set_material(projectname, designname, solutiontype="Eigenmode", **kwargs):
        calls["project_name"] = projectname
        calls["design_name"] = designname
        calls["solution_type"] = solutiontype
        calls["renderer"] = kwargs.get("renderer")

    monkeypatch.setattr(utils_ansys, "setMaterialProperties", fake_set_material)

    class FakeEPR:
        def __init__(self):
            self.sim = SimpleNamespace(_analyze=lambda: None)

        def get_frequencies(self):
            return SimpleNamespace(values=[[5.0, 100.0]])

    renderer = SimpleNamespace(pinfo=SimpleNamespace(project_name="Project1", design_name="CavitySweep_hfss"))

    freq, quality_factor, kappa = utils_ansys.get_freq_Q_kappa(FakeEPR(), renderer)

    assert freq == 5.0e9
    assert quality_factor == 100.0
    assert kappa == 5.0e7
    assert calls == {
        "project_name": "Project1",
        "design_name": "CavitySweep_hfss",
        "solution_type": "Eigenmode",
        "renderer": renderer,
    }
