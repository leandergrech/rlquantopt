import base64
import io
import json
import os
import zlib
from datetime import datetime

from krotov.functionals import F_avg
from scipy.interpolate import CubicSpline
import numpy as np


def _serialize_and_encode(data, serializer, compress=True, **kwargs):
	with io.BytesIO() as buff:
		serializer(buff, data, **kwargs)
		buff.seek(0)
		serialized_data = buff.read()

	if compress:
		serialized_data = zlib.compress(serialized_data)
	return base64.standard_b64encode(serialized_data).decode()


def _decode_and_deserialize(data, deserializer, decompress=True):
	decoded = base64.standard_b64decode(data)
	if decompress:
		decoded = zlib.decompress(decoded)

	with io.BytesIO() as buff:
		buff.write(decoded)
		buff.seek(0)
		return deserializer(buff)


class Encoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, complex):
			return {"__type__": "complex", "__value__": [obj.real, obj.imag]}
		if isinstance(obj, np.ndarray):
			if obj.dtype == object:
				return {"__type__": "ndarray", "__value__": obj.tolist()}
			value = _serialize_and_encode(obj, np.save, allow_pickle=False)
			return {"__type__": "ndarray", "__value__": value}
		if hasattr(obj, "to_json"):
			return {"__type__": "to_json", "__value__": obj.to_json()}


def object_hook(obj):
	if "__type__" in obj:
		obj_type = obj["__type__"]
		obj_val = obj["__value__"]

		if obj_type == "complex":
			return obj_val[0] + 1j * obj_val[1]
		if obj_type == "ndarray":
			if isinstance(obj_val, list):
				return np.array(obj_val)
			return _decode_and_deserialize(obj_val, np.load)
		if obj_type == "to_json":
			return obj_val

	return obj


class Decoder(json.JSONDecoder):
	def __init__(self, *args, **kwargs):
		super().__init__(object_hook=object_hook, *args, **kwargs)


def save_result(data, folder):
	os.makedirs(folder, exist_ok=True)
	with open(f'{folder}/{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.json', 'w') as f:
		json.dump(data, f, indent=4, cls=Encoder)


def cubic_spline_with_zeroes(params, time_slots):
	_params = np.copy(params)
	_params = np.insert(_params, 0, 0)
	_params = np.append(_params, 0)
	c = CubicSpline(time_slots, _params)

	return c


def gate_fidelity(params, tlist, time_slots, H0, H1):
	time_slots = np.linspace(0, tlist[-1], (len(params) + 2))
	H = [H0, [H1, cubic_spline_with_zeroes(params, time_slots)(tlist)]]

	args_list = [(H, state, tlist) for state in full_liouville_basis]

	results = [self.wrapped_mesolve(args) for args in args_list]

	return F_avg(results, self.basis_states, self.unitary, prec=1e-4)
