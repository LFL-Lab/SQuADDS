.. SQuADDS documentation: the landing page, set as the title block of a paper.

.. _welcome-to-squadds-s-documentation:

.. rst-class:: paper-landing-title

SQuADDS
=======

.. raw:: html

   <div class="paper-titleblock">
     <p class="paper-sub">A validated design database and simulation workflow for superconducting qubits</p>
     <p class="paper-abstract">SQuADDS gives you a well-characterised starting point for superconducting hardware. Ask for the Hamiltonian you want and get the closest pre-simulated designs from an experimentally validated database, an interpolated best guess, and, from version 1.0, the full capacitance matrix of any planar layout straight from its GDS file.</p>
     <div class="paper-install">
       <div class="paper-cmd"><pre><code>pip install SQuADDS</code></pre><button class="paper-copy" type="button" data-copy="pip install SQuADDS">Copy</button></div>
       <p class="paper-install__alt">Python 3.11 or newer. For the capacitance models: <code class="docutils literal notranslate"><span class="pre">pip install "squadds[ml]"</span></code>. From source with <code class="docutils literal notranslate"><span class="pre">uv sync</span></code>: <a href="source/getting_started.html">installation guide</a>.</p>
     </div>
     <nav class="paper-links paper-links--row" aria-label="SQuADDS resources"><a href="https://huggingface.co/collections/SQuADDS/squadds-datasets-6a65152b857f8efa5620880a"><img src="_static/images/hf-logo.svg" alt="" width="22" height="22"><span>Datasets</span></a><a href="https://huggingface.co/spaces/SQuADDS/squadds-ml-inference-api"><svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M8 4c-2 0-3 1-3 3v2c0 1.5-.8 2.5-2 3 1.2.5 2 1.5 2 3v2c0 2 1 3 3 3"/><path d="M16 4c2 0 3 1 3 3v2c0 1.5.8 2.5 2 3-1.2.5-2 1.5-2 3v2c0 2-1 3-3 3"/><circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none"/></svg><span>Inference API</span></a><a href="https://deepwiki.com/LFL-Lab/SQuADDS"><svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16v11H9l-5 4z"/><path d="M8 9.5h8M8 12.5h5"/></svg><span>Chat with SQuADDS</span></a></nav>
   </div>

.. _overview:

.. _squadds-a-superconducting-qubit-and-device-design-and-simulation-database:

Where to start
--------------

.. container:: paper-paths

   .. container:: paper-path

      .. raw:: html

         <div class="paper-path__text">
           <p class="paper-who">Qubit designers</p>
           <h3><a href="source/tutorials/Tutorial-1_Getting_Started_with_SQuADDS.html">Find a design for your Hamiltonian</a></h3>
           <p class="paper-desc">Give target parameters, get the closest pre-simulated designs and an interpolated best guess to simulate.</p>
           <p class="paper-go"><a href="source/tutorials/Tutorial-1_Getting_Started_with_SQuADDS.html">Tutorial 1</a><a href="source/tutorials/Tutorial-2_Simulate_interpolated_designs.html">Tutorial 2</a></p>
         </div>

      .. code-block:: python

         db = SQuADDS_DB()
         db.select_system("qubit")
         db.select_qubit("TransmonCross")
         db.create_system_df()
         Analyzer(db).find_closest(
             {"qubit_frequency_GHz": 4,
              "anharmonicity_MHz": -200}, num_top=3)

   .. container:: paper-path

      .. raw:: html

         <div class="paper-path__text">
           <p class="paper-who">Layout to capacitance</p>
           <h3><a href="source/ml_models.html">Capacitance matrix from a GDS file</a></h3>
           <p class="paper-desc">The full N&#8202;&times;&#8202;N Maxwell matrix of every conductor, ground included, in about a second on a CPU. Nets are never merged.</p>
           <p class="paper-go"><a href="source/ml_models.html">ML models guide</a><a href="source/tutorials/Tutorial-11_Predicting_Capacitance_Matrices_from_GDS.html">Tutorial 11</a></p>
         </div>

      .. code-block:: python

         predictor = CapacitancePredictor(
             load_model("QCap"))
         res = predictor.predict("chip.gds")
         res.to_dataframe()   # fF, ground first

   .. container:: paper-path

      .. raw:: html

         <div class="paper-path__text">
           <p class="paper-who">Contributors</p>
           <h3><a href="source/resources/contribute.html">Add your designs and data</a></h3>
           <p class="paper-desc">Simulated or measured devices, capacitance data and models. The validator checks every matrix before a pull request opens.</p>
           <p class="paper-go"><a href="https://squadds-portal.vercel.app/">Contribution portal</a><a href="source/tutorials/Tutorial-3_Contributing_Validated_Simulation_Data_to_SQuADDS.html">Tutorial 3</a><a href="source/tutorials/Tutorial-15_Contributing_Capacitance_Data_and_Models.html">Tutorial 15</a></p>
         </div>

      .. code-block:: python

         validate_capacitance_contribution(
             "my_batch/")
         contribute_capacitance("my_batch/")

   .. container:: paper-path

      .. raw:: html

         <div class="paper-path__text">
           <p class="paper-who">AI agents</p>
           <h3><a href="source/mcp_server.html">Connect an assistant with MCP</a></h3>
           <p class="paper-desc">The SQuADDS MCP server lets a coding assistant search designs, find the closest match and interpolate parameters.</p>
           <p class="paper-go"><a href="source/mcp_server.html">MCP server guide</a><a href="https://deepwiki.com/LFL-Lab/SQuADDS">Chat with the codebase</a></p>
         </div>

      .. code-block:: bash

         uv sync --extra mcp
         uv run squadds-mcp

Learn more about our latest paper
---------------------------------

.. raw:: html

   <figure class="paper-video">
     <video controls preload="none" playsinline poster="ml-paper/media/v1_summary.jpg" width="1280" height="720" aria-label="Summary video of the paper"><source src="ml-paper/media/v1_summary.mp4" type="video/mp4"></video>
     <figcaption><a href="ml-paper/">QCap: a physics-corrected graph neural network for the capacitance matrix of any planar superconducting device layout</a></figcaption>
   </figure>

Tutorials
---------

.. rst-class:: paper-tutorials

0. :doc:`Using the SQuADDS Web UI <source/tutorials/Tutorial-0_Using_the_SQuADDS_WebUI>`
1. :doc:`Getting started with SQuADDS <source/tutorials/Tutorial-1_Getting_Started_with_SQuADDS>`
2. :doc:`Simulating interpolated designs <source/tutorials/Tutorial-2_Simulate_interpolated_designs>`
3. :doc:`Contributing validated simulation data <source/tutorials/Tutorial-3_Contributing_Validated_Simulation_Data_to_SQuADDS>`
4. :doc:`Contributing measured data <source/tutorials/Tutorial_4_Contributing_Measured_Data_to_SQuADDS>`
5. :doc:`Designing a fab-ready chip <source/tutorials/Tutorial-5_Designing_a_fab_ready_chip_with_SQuADDS>`
6. :doc:`Adding airbridges <source/tutorials/Tutorial-6_Adding_Airbridges>`
7. :doc:`Simulating designs with Palace <source/tutorials/Tutorial-7_Simulate_designs_with_palace>`
8. :doc:`Machine learning with SQuADDS <source/tutorials/Tutorial-8_ML_interpolation_in_SQuADDS>`
9. :doc:`Driven-modal capacitance extraction <source/tutorials/Tutorial-9_DrivenModal_Capacitance_Extraction>`
10. :doc:`Unified driven-modal Hamiltonian extraction <source/tutorials/Tutorial-10_DrivenModal_Combined_Hamiltonian_Extraction>`
11. :doc:`Predicting capacitance matrices from GDS <source/tutorials/Tutorial-11_Predicting_Capacitance_Matrices_from_GDS>`
12. :doc:`Planar capacitance families <source/tutorials/Tutorial-12_Planar_Capacitance_Families_in_SQuADDS>`
13. :doc:`From layout to Hamiltonian with ML <source/tutorials/Tutorial-13_From_Layout_to_Hamiltonian_with_ML>`
14. :doc:`Training on your own designs <source/tutorials/Tutorial-14_Training_on_Your_Own_Designs>`
15. :doc:`Contributing capacitance data and models <source/tutorials/Tutorial-15_Contributing_Capacitance_Data_and_Models>`
16. :doc:`Inverse design by ML screening <source/tutorials/Tutorial-16_Inverse_Design_by_ML_Screening>`

.. _quick-links:

.. _further-information:

Contents
--------

.. rst-class:: paper-contents

- :doc:`Getting started <source/getting_started>`: set up SQuADDS and start using it.
- :doc:`Tutorials <source/tutorials/index>`: step-by-step guides for common workflows.
- :doc:`MCP server <source/mcp_server>`: connect AI coding assistants to the SQuADDS database.
- :doc:`ML models <source/ml_models>`: planar capacitance models and the hosted inference models on Hugging Face.
- :doc:`Run ML models locally <source/ml_local>`: the capacitance models on your own computer.
- :doc:`API reference <source/apidocs/index>`: the full API documentation.
- `Contribution portal <https://squadds-portal.vercel.app/>`_: contribute to SQuADDS in seconds.
- :doc:`Contributing <source/resources/contribute>`: get involved in the development of SQuADDS.
- :doc:`Developer notes <source/developer/index>`: insights and guidelines for developers.
- :doc:`Simulation notes <source/resources/simulations>`: how our simulations are set up.
- :doc:`Workflows <source/resources/concepts>`: the key workflows within SQuADDS.
- :doc:`Release notes <source/release_notes>`: updates and new releases.
- `Chat with the codebase <https://deepwiki.com/LFL-Lab/SQuADDS>`_: ask DeepWiki about the code.
- Indices: :ref:`genindex`, :ref:`modindex`, :ref:`search`.

.. toctree::
   :hidden:

   Installation <source/getting_started>
   Tutorials <source/tutorials/index>
   MCP Server <source/mcp_server>
   ML Models <source/ml_models>
   Run ML Models Locally <source/ml_local>
   API Reference <source/apidocs/index>
   Contributions <source/resources/contribute>
   Developer Notes <source/developer/index>
   Simulation Notes <source/resources/simulations>
   Workflows <source/resources/concepts>
   Release Logs<source/release_notes>
   References<source/references/index>
   GitHub <https://github.com/lfl-lab/SQuADDS>
   Paper <https://arxiv.org/pdf/2312.13483.pdf>
   LFL Lab ✌️<https://dornsife.usc.edu/lfl/>

.. _citation:

Citing SQuADDS
--------------

If SQuADDS helps your research, please support its continued development and maintenance by citing the paper: Shanto *et al.*, *Quantum* **8**, 1465 (2024).

.. code-block:: latex

   @article{Shanto2024squaddsvalidated,
     doi = {10.22331/q-2024-09-09-1465},
     url = {https://doi.org/10.22331/q-2024-09-09-1465},
     title = {{SQ}u{ADDS}: {A} validated design database and simulation workflow for superconducting qubit design},
     author = {Shanto, Sadman and Kuo, Andre and Miyamoto, Clark and Zhang, Haimeng and Maurya, Vivek and Vlachos, Evangelos and Hecht, Malida and Shum, Chung Wa and Levenson-Falk, Eli},
     journal = {{Quantum}},
     issn = {2521-327X},
     publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
     volume = {8},
     pages = {1465},
     month = sep,
     year = {2024}
   }
