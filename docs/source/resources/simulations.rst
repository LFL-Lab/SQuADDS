Simulation Lessons
==================

This is a guide to the simulation methodologies we have found to be useful in our project.

Palace Simulations
-------------------

- **Installation Guide:** :doc:`palace`.

.. note::

   Section Under Construction 🛠️


ANSYS Simulations
-----------------

We have found the following practices to be effective in ANSYS simulations:

- Finer mesh is always better, but as long as the seed mesh is no more than 1/3 the size of a feature it spans, it will converge to a good adaptive mesh.

- The convergence criteria (Delta-S or Delta-f, depending on the solver) should be less than 0.02%; 0.01% is better. The default (0.2%) never works.

- There should be a minimum of 1 additional simulation pass after the convergence criteria are met; more is better (but slower).

- EPR analysis has been quite unreliable for getting coupling strengths, with typical errors above 30%. We use a lumped approach similar to LOM whenever possible, and only use eigenmodal analysis for frequencies and quality factors.

- We get best results simulating small components separately from each other.

- Resonator quality factor is never super accurate, but the best approach we've found is to do an eigenmodal analysis of the resonator and feedline (with feedline finely meshed and extending far beyond the coupling region), with the feedline terminated by 50-ohm lumped ports.

- Each mesh cell can be represented by a constant, linear function, or 2nd-order polynomial. We've found best results with a linear function.

- *Not strictly simulations, but important for analysis:* The rotating wave approximation can cause one to mis-estimate dispersive shifts by a factor of ~2 if you have far-detuned qubit and cavity.

Resources
=========

Some learning resources we've found useful:

- Helpful EM Simulation Resources:
    - `CPW Resonator Simulations <https://smm.misis.ru/CPW-resonator-coupling/>`_
    - `Various Microwave Calculator Tools <https://www.microwaves101.com/calculators>`_

- Ansys Learning Resources:
    - `HowToSim YouTube Channel <https://youtube.com/@howtosim7253?feature=shared>`_

    - `HFSS Tutorial for Axion Cavity Workshop (PDF) <https://indico.fnal.gov/event/13068/contributions/17083/attachments/11439/14607/MJones_-_HFSS_Tutorial_for_Axion_Cavity_Workshop.pdf>`_

    - `Ansys Courses <https://courses.ansys.com/index.php/electronics/>`_

    - `Advanced Meshing Techniques (PDF) <http://www.ece.uprm.edu/~rafaelr/inel6068/HFSS/3570_Advanced_Meshing_Techniques.pdf>`_

    - `Simulation Guides <https://github.com/McDermott-Group/Simulation-and-Design/tree/master/Simulation%20Guides>`_

- Ansys Scripting Resources:
    - `Ansys Scripting Forum <https://forum.ansys.com/categories/scripting>`_

    - `GitHub Repository for HowToSim <https://github.com/linmingchih>`_

    - `HFSS Library <https://arrc.ou.edu/~cody/hfsslib/>`_

    - `PyANSYS Documentation <https://aedt.docs.pyansys.com/version/stable/>`_

- Ansys Parallel Jobs:
    - `Using ANSYS RSM <https://www.hpc.iastate.edu/guides/using-ansys-rsm>`_



Get Involved
============

We continuously enhance our simulation techniques and strive to maintain the most current information on this webpage. We're eager to understand how closely our methods align with your team's practices and to learn about any challenges you've encountered previously. We welcome your insights and feedback, so please feel free to reach out to us!
