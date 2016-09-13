************
Introduction
************


Overview of Osmos' project
****************************
The current project aims at analyse the data collected by apparatus developped by Osmos, and can be viewed as a problem of `structure health monitoring <https://en.wikipedia.org/wiki/Structural_health_monitoring>`_ (SHM). Data provided by Osmos are collected using a distributed array of sensors installed at different positions of a structure, and each one records the local value of the temperature and the elongation (of an optic cord). Data are composed of two types:

   * static data with the sampling period of 1 hour
   * dynamic data which are triggered by some (non-thermal) environmental events with the sampling period of 20 ms.

The duration of dynamic data are generally very short hence does not depend on the variation of the temperature.
.. The two types of data may be dependent: certain environmental events in the dynamic data may result elastic deformation of the structure and change consequently the static data.

Objective
---------
The ultimate goal is to detect structural changes which are not caused by environmental factors. More precisely, on static data this means to reveal the changes of elongation by removing the contribution of thermal expansion or contraction.

.. todo:: The study of the dynamic data will be tackled in a future version.

Thermal expansion coefficient
=============================
In the general case of a gas, liquid, or solid, the `volumetric coefficient of thermal expansion <https://en.wikipedia.org/wiki/Thermal_expansion>`_ is given by:

.. math::
   \alpha \Delta T = \frac {V} {\Delta V}
.. \left(\frac{\partial V}{\partial T}\right)

This is the theoretical relationship between the volume and the temperature, which states that the changes in the temperature is the cause of the (relative) changes in the volume and the two are proportional. Clearly using the thermal law to describe the changes in volume will deviate from reality when the non-thermal contributions, which are not measured by apparatus of Osmos, become dominant.

A simple linear model with thermal delay
----------------------------------------
In the regime that :math:`V=V_0+\Delta V` and :math:`\Delta V / V_0\ll 1` (changes of volume is small compared to the characteristic volume), one can approximate the relation above by

.. math::
   \Delta V \propto \alpha \Delta T

The estimation of the coefficient :math:`\alpha` can be done via linear regression: one writes for this effect

.. math::
   V'(t) = \alpha T'(t).

Simlilar relation holds if one replace the measurement of volume by that of length, as it is the case of the data provided by Osmos.

The *thermal delay*, which is the time takes the structure to react to the temperature variation, must be taken into account in this estimation. More precisely, let :math:`x_t, y_t` denote the temperature and elongation (or their derivatives) of instant :math:`t`, the linear model reads

.. math::
   y_t =  \alpha x_{t-\Delta \tau} + \beta

With the estimated coefficients one can use this model to determine the thermal contribution in the observed elongation. However this model does not fit well data, moreover the regression of :math:`\alpha, \beta` depends on the estimation of :math:`\Delta \tau` which can only takes discrete values due to the sampling hence introduce important errors.

Methodology
===========

Input-Ouput or Output-Only analysis
-----------------------------------
A visual inspection on the static data provided by Osmos reveals some important types of behavior:

   #. On some sensors the signals present remarkable daily variations (of period of 24 hours), apparently due to either air conditionning or natural daily variation of the temperature. On these types of data we talk about *Input-Output* analysis since we try to explain the elongation by the temperature and the linear model mentioned above may be applicable.
   #. On some sensors the temperature looks like piece-wise constant and uncorrelated with the elongation. We talk then about *Output-Only* analysis since apparently the temperature does not contribute to the elongation which is the only useful information (possibly caused by some unknown environmental factors).
Note that we also observed other types signals that are considered as errorneous and inexploitable, *e.g.* the elongation contains some noise-like passages due to possible failure of a sensor.

Decomposition of Signals
~~~~~~~~~~~~~~~~~~~~~~~~
This overly simplified classfication of course does not resume all the characteristics of the data and the linear temperature-elongation relation seems more the exception than the rule in most situations. For example, on the first type of data the signals can generally be decomposed into two components:

   #. Seasonal component (or rapid component) : daily variation of the signal such as the temperature cycle, highly periodic. On this component the temperature seems to be highly correlated to the elongation.
   #. Trend component (or slow component) : what remains once the seasonal component is removed. On this component the temperature seems to be uncorrelated to the elongation in most situations.
Hence a linear model may work well on the seasonal component but fail on the trend component.

The approach proposed by Sivienn, in the current state of development, is to combine different models working on possibly distinct components of data.


Multiple alarms
---------------




   * Input-Output analysis:
   * Output-Only analysis:

relies also on a linear model which can be seen as a generalisation of the model above.


Removal of the amounts to explain the elongation

In the ideal case,


concerns only the static data


Overview of the library Pyshm
*******************************
This library provides some tools of analysis for the following problem(s) encountered in


Besides the library itself, Pyshm is shipped with some command-line scripts.

.. and a web-based interface.


The strategy taken here is to

.. math::
   y_t = \sum_{i=1}^p h_i y_{t-i} + \sum_{j=0}^{q-1} g_j x_{t-j} + c_t + \varepsilon_t

   y_t =  \alpha x_{t-\Delta T} + \beta

   y_t = \sum_{j=0}^{p-1} a_j x_{t-j} + \epsilon_t
