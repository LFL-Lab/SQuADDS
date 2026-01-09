SQuADDS Workflows
=================

From Hamiltonian to GDS File: APS March Meeting Talk
----------------------------------------------------

This talk given by Sadman Ahmed Shanto at the `APS March Meeting 2024 - Session A47.8 <https://meetings.aps.org/Meeting/MAR24/Session/A47.8>`_ introduces the SQuADDS project and goes over an example workflow from Hamiltonian to GDS file.

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/0bBYAHgYPzc" title="SQuADDS" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



SQuADDS Basics by Anne and Jenny
----------------------------------

The SQuADDS database provides information on datasets of different components and the data available regarding the components. You may query data for specific components using the name of the component and its data type, and the SQuADDS database returns various results of simulated designs of the component.

**Creation of a System of Interest**

The SQuADDS API provides a useful method to query the system for desired components, create your new system, and search for the closest designs to your desired Hamiltonian parameters.

**Initial Creation of a System:**

First, you must access the SQuADDS database by creating a database object - `db = SQuADDS_DB()`.

**Selection of Components:**

You can select the system using the ``select_system()`` function, providing the desired components by their names.

For a qubit and cavity claw configuration, you can call a single ``select_system()`` method, with the names ``“qubit”`` and ``“cavity_claw”`` as an array.

Example:

.. code-block:: python

    db.select_system(["qubit", "cavity_claw"])

You can select specific types of these components such as ``select_qubit(“TransmonCross”)`` and ``select_cavity_claw(“RouteMeander”)`` with the name of the specific component as a parameter.

**Merging Components Together to Create a System:**

In order to check if the system is valid, you can use the ``show_selections()`` function, which returns the values that were selected from the SQuADDS database. To combine the components, you can call ``create_system_df()``.

**Querying for Designs:**

You can pass the SQuADDS database object you have created from selecting the different components as a parameter of the ``Analyzer`` object and create a new analyzer object. The ``Analyzer`` object contains a function, ``.find_closest()``, that allows you to query for designs to your system, by inserting your target parameters. Target parameters are defined by the Hamiltonian parameters of the system which includes qubit frequency, cavity frequency, kappa, resonance type, anharmonicity and coupling strength.

After calling the ``.find_closest()`` function on your analyzer object with your desired input parameters, the SQuADDS API will search for simulated designs for your desired system and Hamiltonian parameters.

Another useful function from the ``Analyzer`` object is the ``.closest_design_in_H_space()`` method, which produces a graph visualization displaying where the closest design lies in the Hamiltonian parameter space in relation to the locations of all pre-simulated designs and the target design.

**Interpolation of Designs:**

The SQuADDS API includes an interpolation object called ``ScalingInterpolator`` to allow you to interpolate your designs. ``ScalingInterpolator`` uses a physics based interpolation algorithm using your target parameters and the analyzer to interpolate the most optimal design.

You can create a ``ScalingInterpolator`` object by passing in your target parameters and your analyzer object. To interpolate the designs, you can call the function ``.get_design()`` to produce.
