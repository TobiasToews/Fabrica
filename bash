Traceback (most recent call last):
  File "/home/tobi/Fabrica/rendering/render_motion_plan.py", line 239, in <module>
    render_motion_plan(args.assembly_dir, args.log_dir, 
  File "/home/tobi/Fabrica/rendering/render_motion_plan.py", line 115, in render_motion_plan
    arm_chain_move = get_arm_chain(arm_type, 'move')
  File "/home/tobi/Fabrica/planning/robot/util_arm.py", line 57, in get_arm_chain
    arm_chain = get_panda_arm_chain(base_pos, base_euler, reduced_limit=reduced_limit)
  File "/home/tobi/Fabrica/planning/robot/util_arm.py", line 24, in get_panda_arm_chain
    chain = Chain.from_urdf_file(os.path.join(project_base_dir, 'assets/panda/panda.urdf'), base_elements=['panda_link0'],
  File "/home/tobi/Fabrica/planning/robot/ik/chain.py", line 216, in from_urdf_file
    links = get_urdf_parameters(urdf_file, base_elements=base_elements, last_link_vector=last_link_vector, base_element_type=base_element_type, symbolic=symbolic, scale_translation=scale_translation,
  File "/home/tobi/Fabrica/planning/robot/ik/urdf.py", line 151, in get_urdf_parameters
    parameters.append(lib_link.URDFLink(
  File "/home/tobi/miniconda3/envs/fabrica/lib/python3.10/site-packages/ikpy/link.py", line 158, in __init__
    self.symbolic_transformation_matrix = self._apply_geometric_transformations(theta=theta, mu=mu, symbolic=True)
  File "/home/tobi/miniconda3/envs/fabrica/lib/python3.10/site-packages/ikpy/link.py", line 233, in _apply_geometric_transformations
    symbolic_frame_matrix = sympy.lambdify([theta, mu], symbolic_frame_matrix, "numpy")
  File "/home/tobi/miniconda3/envs/fabrica/lib/python3.10/site-packages/sympy/utilities/lambdify.py", line 942, in lambdify
    expr_str = str(expr)
  File "/home/tobi/miniconda3/envs/fabrica/lib/python3.10/site-packages/sympy/matrices/matrixbase.py", line 3725, in __str__
    return "Matrix(%s)" % str(self.tolist())
  File "/home/tobi/miniconda3/envs/fabrica/lib/python3.10/site-packages/sympy/core/_print_helpers.py", line 29, in __str__
    return sstr(self, order=None)
  File "/home/tobi/miniconda3/envs/fabrica/lib/python3.10/site-packages/sympy/printing/printer.py", line 408, in __call__
    return self.__wrapped__(*args, **kwargs)
  File "/home/tobi/miniconda3/envs/fabrica/lib/python3.10/site-packages/sympy/printing/str.py", line 992, in sstr
    p = StrPrinter(settings)
  File "/home/tobi/miniconda3/envs/fabrica/lib/python3.10/site-packages/sympy/printing/printer.py", line 266, in __init__
    self._settings.update(settings)
KeyboardInterrupt
