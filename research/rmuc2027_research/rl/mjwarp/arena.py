"""Live Arena substep bridge. CPU rules/queries, MJWarp rigid-body integration."""
import json
import hashlib
from collections import OrderedDict
import math
import time
import xml.etree.ElementTree as ET
from .runtime import BatchPhysics, dependencies


_SIGNATURE_CACHE = OrderedDict()

MODEL_FIELDS = ('geom_pos', 'geom_quat', 'geom_size', 'geom_aabb', 'geom_rbound', 'body_mass', 'body_ipos', 'body_iquat', 'body_inertia', 'body_invweight0')


def text(values):
    return ' '.join(str(float(v)) for v in values)


def contact_geometry_ids(contact, ngeom):
    """MJWarp fills mjContact.geom, not the independent legacy geom1/geom2.

    MuJoCo 3.17 still exposes those legacy fields, but get_data_into leaves
    them untouched. Reading them can report random IDs or the wrong body.
    """
    ids = tuple(int(g) for g in contact.geom)
    if len(ids) != 2 or any(g < 0 or g >= ngeom for g in ids):
        raise RuntimeError(f'Invalid MJWarp contact.geom {ids}; model has {ngeom} geometries')
    return ids


class ArenaPhysics:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.runtime = None
        self.topology = None
        self.world_key = object()
        self.collider_cache = {}
        self.geometry_cache = OrderedDict()
        self.geometry_cache_limit = 1024
        self.mj, _, self.np, _ = dependencies()

    def multiply(self, a, b):
        out = self.np.empty(4)
        self.mj.mju_mulQuat(out, self.np.array(a, dtype=float), self.np.array(b, dtype=float))
        return out

    def rotate(self, vector, quat):
        out = self.np.empty(3)
        self.mj.mju_rotVecQuat(out, self.np.array(vector, dtype=float), self.np.array(quat, dtype=float))
        return out

    def flatten(self, shape, pos, quat):
        kind, data = next(iter(shape.items()))
        if kind == 'Compound':
            for transform, child in data['shapes']:
                q = transform['rotation']; q = [q[3], *q[:3]]
                yield from self.flatten(child, self.np.array(pos)+self.rotate(transform['translation'], quat), self.multiply(quat, q))
            return
        if kind == 'RoundCuboid':
            h = data['inner_shape']['half_extents']; r = data['border_radius']
            # Exact Minkowski sum: three expanded slabs, twelve edge capsules,
            # and eight corner balls. All pieces belong to the same rigid body.
            for axis in range(3):
                size = list(h); size[axis] += r
                yield ('box', pos, quat, size, None)
            for axis in range(3):
                other = [i for i in range(3) if i != axis]
                direction = self.np.zeros(3); direction[axis] = 1
                qedge = self.np.empty(4); self.mj.mju_quatZ2Vec(qedge, direction)
                for u in (-1, 1):
                    for v in (-1, 1):
                        offset = self.np.zeros(3); offset[other[0]] = u*h[other[0]]; offset[other[1]] = v*h[other[1]]
                        yield ('capsule', self.np.array(pos)+self.rotate(offset, quat), self.multiply(quat, qedge), [r, h[axis]], None)
            return
        if kind == 'Cuboid':
            yield ('box', pos, quat, data['half_extents'], None)
        elif kind == 'Ball':
            yield ('sphere', pos, quat, [data['radius']], None)
        elif kind == 'Cylinder':
            # Rapier cylinders point along Y; MuJoCo cylinders point along Z.
            yield ('cylinder', pos, self.multiply(quat, [math.sqrt(.5), math.sqrt(.5), 0, 0]), [data['radius'], data['half_height']], None)
        elif kind == 'Capsule':
            a, b = data['segment']['a'], data['segment']['b']
            delta = self.np.array(b)-a; q = self.np.empty(4)
            self.mj.mju_quatZ2Vec(q, delta)
            center = (self.np.array(a)+b)/2
            yield ('capsule', self.np.array(pos)+self.rotate(center, quat), self.multiply(quat, q), [data['radius'], float(self.np.linalg.norm(delta))/2], None)
        elif kind == 'TriMesh':
            vertices = self.np.asarray(data['vertices'])
            floor = float(vertices[:, 2].min())
            # Terrain is extruded vertically by the original engine. Decompose
            # its upper faces into convex triangular prisms, preserving concavity.
            for face in data['indices']:
                top = vertices[face]
                normal = self.np.cross(top[1]-top[0], top[2]-top[0])
                if normal[2] <= 1e-10:
                    continue
                bottom = top.copy(); bottom[:, 2] = min(floor, float(top[:, 2].min())-.001)
                yield ('mesh', pos, quat, [], self.np.concatenate((top, bottom)).tolist())
        else:
            raise ValueError(f'Unsupported Arena collision shape: {kind}')

    def geometry(self, request):
        result = []
        for collider in request['colliders']:
            if collider['sensor']:
                continue
            # Collider numbers repeat across worlds. Cache immutable descriptors
            # by identity and retain their owner to prevent Python id reuse.
            key = id(collider)
            cached = self.geometry_cache.get(key)
            if cached is None:
                p = collider['pose']
                pieces = [(f"c{collider['id']}_{j}", collider, geom) for j, geom in enumerate(self.flatten(collider['shape'], p['pos'], p['quat']))]
                cached = self.geometry_cache[key] = (collider, pieces)
                if len(self.geometry_cache) > self.geometry_cache_limit:
                    self.geometry_cache.popitem(last=False)
            self.geometry_cache.move_to_end(key)
            result.extend(cached[1])
        return result

    def build_xml(self, request, geometry):
        root = ET.Element('mujoco', model='rmuc2027')
        ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='false')
        ET.SubElement(root, 'option', timestep=str(request['dt']), gravity='0 0 -9.81', integrator='Euler', solver='Newton', iterations='20')
        assets = ET.SubElement(root, 'asset'); world = ET.SubElement(root, 'worldbody')
        parents = {None: world}
        for body in request['bodies']:
            p = body['pose']; attrs = dict(name=f"b{body['id']}", pos=text(p['pos']), quat=text(p['quat']))
            if not body['dynamic']:
                attrs['mocap'] = 'true'
            node = parents[body['id']] = ET.SubElement(world, 'body', **attrs)
            if body['dynamic']:
                ET.SubElement(node, 'freejoint', name=f"j{body['id']}")
                mp = body['mass_properties']; q = mp['principal_inertia_local_frame']
                inertia = [1/max(v, 1e-8)**2 for v in mp['inv_principal_inertia_sqrt']]
                ET.SubElement(node, 'inertial', pos=text(mp['local_com']), quat=text([q[3], *q[:3]]), mass=str(max(body['mass'], 1e-5)), diaginertia=text(inertia))
        for name, col, (kind, pos, quat, size, vertices) in geometry:
            attrs = dict(name=name, type=kind, pos=text(pos), quat=text(quat), friction=f"{col['friction']} .005 .0001", contype='1' if col['enabled'] else '0', conaffinity='1' if col['enabled'] else '0')
            if kind == 'mesh':
                ET.SubElement(assets, 'mesh', name=name, vertex=text([x for v in vertices for x in v]))
                attrs['mesh'] = name
            else:
                attrs['size'] = text(size)
            ET.SubElement(parents[col['body']], 'geom', **attrs)
        return ET.tostring(root, encoding='unicode')

    def compile(self, request, geometry, count=1):
        xml = self.build_xml(request, geometry)
        if self.runtime:
            self.runtime.close()
        for name in ('_contact_ids', '_contact_forces'):
            self.__dict__.pop(name, None)
        self.runtime = BatchPhysics(xml, count, self.device, request['dt'], nconmax=2048, njmax=4096)
        with self.runtime.wp.ScopedDevice(self.runtime.device):
            self.runtime.model = self.runtime.mjw.put_model(self.runtime.cpu_model,
                batch_sizes={name: count for name in MODEL_FIELDS})

        model = self.runtime.cpu_model
        self.host_models = [{name: self.np.array(getattr(model, name), copy=True) for name in MODEL_FIELDS} for _ in range(count)]
        self.host_data = [self.mj.MjData(model) for _ in range(count)]
        self.host_states = OrderedDict()
        self.initial_model = {name: self.np.array(getattr(model, name), copy=True) for name in MODEL_FIELDS}
        self.prepared_geometry = [{} for _ in range(count)]
        self.geom_ids = {self.mj.mj_id2name(model, self.mj.mjtObj.mjOBJ_GEOM, i): i for i in range(model.ngeom)}
        self.body_ids = {self.mj.mj_id2name(model, self.mj.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}

    @staticmethod
    def signature(request):
        # Collision masks are not batchable in MJWarp; worlds with differing
        # active covers are grouped into separate model batches.
        def stripped(shape):
            kind, data = next(iter(shape.items()))
            if kind == 'TriMesh':
                return [kind, data['vertices'], data['indices']]
            if kind == 'Compound':
                return [kind, [stripped(child) for _, child in data['shapes']]]
            return kind
        signatures = []
        for c in request['colliders']:
            key = id(c)
            cached = _SIGNATURE_CACHE.get(key)
            if cached is None:
                descriptor = (c['id'], c['body'], c['enabled'], c['pose'] if 'TriMesh' in c['shape'] else None, stripped(c['shape']))
                digest = hashlib.sha256(json.dumps(descriptor, separators=(',', ':')).encode()).digest()
                cached = _SIGNATURE_CACHE[key] = (c, digest)
                if len(_SIGNATURE_CACHE) > 4096:
                    _SIGNATURE_CACHE.popitem(last=False)
            _SIGNATURE_CACHE.move_to_end(key)
            signatures.append(cached[1])
        return tuple(signatures)

    def step(self, request):
        if not request.get('colliders_delta', False):
            self.collider_cache = {}
        elif not self.collider_cache:
            raise RuntimeError('Collider delta received before initial scene')
        for identifier in request.get('removed_colliders', []):
            if identifier not in self.collider_cache:
                raise RuntimeError(f'Unknown removed collider {identifier}')
            del self.collider_cache[identifier]
        for collider in request['colliders']:
            self.collider_cache[collider['id']] = collider
        request = {**request, 'colliders': list(self.collider_cache.values()), '_world_key': self.world_key}
        from .batching import active_service
        if active_service is not None:
            return active_service.submit(self.device, request)
        return self.step_many([request])[0]

    def step_many(self, requests):
        self.last_profile = {}
        begin = time.perf_counter()
        geometries = [self.geometry(r) for r in requests]
        signature = (len(requests), self.signature(requests[0]))
        self.last_profile['geometry_signature'] = time.perf_counter()-begin
        begin = time.perf_counter()
        if signature != self.topology:
            self.compile(requests[0], geometries[0], len(requests))
            self.topology = signature
        self.last_profile['compile'] = time.perf_counter()-begin
        begin = time.perf_counter()
        runtime = self.runtime; model = runtime.cpu_model; np = self.np
        cpus = []; model_values = {name: [] for name in MODEL_FIELDS}
        for world, (request, geometry) in enumerate(zip(requests, geometries)):
            cpu = self.prepare(request, geometry, world)
            cpus.append(cpu)
            for name in model_values:
                model_values[name].append(self.host_models[world][name])
        self.last_profile['prepare'] = time.perf_counter()-begin
        begin = time.perf_counter()
        for name, values in model_values.items():
            target = getattr(runtime.model, name)
            target.assign(np.asarray(np.stack(values), dtype=np.float32))
        for name in ('qpos', 'qvel', 'xfrc_applied', 'mocap_pos', 'mocap_quat'):
            runtime._assign(name, np.stack([getattr(cpu, name) for cpu in cpus]))
        runtime._assign('qacc_warmstart', np.zeros((len(requests), model.nv)))
        self.last_profile['upload_enqueue'] = time.perf_counter()-begin
        begin = time.perf_counter()
        runtime.step(np.zeros((len(requests), 0)))
        self.last_profile['step_and_sync'] = time.perf_counter()-begin
        begin = time.perf_counter()
        state = runtime.state()
        self.last_state = state
        self.contacts = self.read_contacts()
        self.last_profile['state_contacts_readback'] = time.perf_counter()-begin
        begin = time.perf_counter()
        results = [self.result(request, cpu, state, world) for world, (request, cpu) in enumerate(zip(requests, cpus))]
        self.last_profile['result_encoding'] = time.perf_counter()-begin
        return results

    def prepare(self, request, geometry, world=0):
        model = self.runtime.cpu_model; np = self.np
        # Arrival order can change between microbatches: prepare caches belong
        # to logical source worlds, never to a transient GPU batch position.
        if '_world_key' in request:
            key = request['_world_key']
            cached = self.host_states.get(key)
            if cached is None:
                cached = self.host_states[key] = (
                    {n: np.array(v, copy=True) for n, v in self.initial_model.items()},
                    self.mj.MjData(model), {})
                if len(self.host_states) > max(4, self.runtime.num_worlds*2):
                    self.host_states.popitem(last=False)
            self.host_states.move_to_end(key)
            self.host_models[world], self.host_data[world], self.prepared_geometry[world] = cached
        for name, values in self.host_models[world].items():
            np.copyto(getattr(model, name), values)
        previous = self.prepared_geometry[world]
        updated = {}
        # Geometry and mass may change as arms move, the turret folds, or robots
        # revive. They are synchronized before integration, never one tick late.
        for name, col, (kind, pos, quat, size, _) in geometry:
            updated[name] = col
            if previous.get(name) is col:
                continue
            i = self.geom_ids[name]
            # Compiled meshes include MuJoCo's centroid/principal-axis offset.
            # Preserve it; their vertices and poses are static within a batch.
            if kind != 'mesh':
                model.geom_pos[i] = pos; model.geom_quat[i] = quat
            if size:
                model.geom_size[i, :len(size)] = size
                if kind == 'box':
                    extent = np.array(size)
                elif kind == 'sphere':
                    extent = np.repeat(size[0], 3)
                elif kind == 'capsule':
                    extent = np.array([size[0], size[0], size[0]+size[1]])
                else:  # cylinder
                    extent = np.array([size[0], size[0], size[1]])
                model.geom_aabb[i, :3] = 0
                model.geom_aabb[i, 3:] = extent
                model.geom_rbound[i] = np.linalg.norm(extent)
            model.geom_contype[i] = model.geom_conaffinity[i] = int(col['enabled'])
        self.prepared_geometry[world] = updated
        if '_world_key' in request:
            old = self.host_states[request['_world_key']]
            self.host_states[request['_world_key']] = (old[0], old[1], updated)
        cpu = self.host_data[world]
        cpu.xfrc_applied.fill(0)
        for b in request['bodies']:
            i = self.body_ids[f"b{b['id']}"]
            pose = b['pose'] if b['dynamic'] else b['next']
            if b['dynamic']:
                mp = b['mass_properties']; iq = mp['principal_inertia_local_frame']
                model.body_mass[i] = max(b['mass'], 1e-5)
                model.body_ipos[i] = mp['local_com']
                model.body_iquat[i] = [iq[3], *iq[:3]]
                model.body_inertia[i] = [1/max(v, 1e-8)**2 for v in mp['inv_principal_inertia_sqrt']]
                # Free-body constraint impedance uses these derived weights.
                # Updating mass/inertia alone leaves the compiled unloaded
                # values in place during leg motion, folding and arm loading.
                model.body_invweight0[i, 0] = 1/model.body_mass[i]
                model.body_invweight0[i, 1] = np.mean(1/model.body_inertia[i])

                joint = model.body_jntadr[i]; q = model.jnt_qposadr[joint]; v = model.jnt_dofadr[joint]
                cpu.qpos[q:q+7] = [*pose['pos'], *pose['quat']]
                inverse = [pose['quat'][0], *(-np.array(pose['quat'][1:]))]
                cpu.qvel[v:v+6] = [*b['velocity'], *self.rotate(b['angular'], inverse)]
                force = np.asarray(b['force']) - b.get('linear_damping', 0)*b['mass']*np.asarray(b['velocity'])
                omega = np.array(b['angular'])
                principal_world = self.multiply(pose['quat'], model.body_iquat[i])
                inverse_principal = [principal_world[0], *(-principal_world[1:])]
                principal_omega = self.rotate(omega, inverse_principal)
                drag = self.rotate(model.body_inertia[i]*principal_omega, principal_world)*b.get('angular_damping', 0)
                cpu.xfrc_applied[i] = [*force, *(np.asarray(b['torque'])-drag)]
            else:
                mocap = model.body_mocapid[i]; cpu.mocap_pos[mocap] = pose['pos']; cpu.mocap_quat[mocap] = pose['quat']
        for name, values in self.host_models[world].items():
            np.copyto(values, getattr(model, name))
        return cpu

    def read_contacts(self):
        """Read one batch of contact data, never the full solver state per world."""
        from mujoco_warp._src.types import ContactType
        r = self.runtime; np = self.np; d = r.data
        count = int(d.nacon.numpy()[0])
        if not 0 <= count <= d.naconmax:
            raise RuntimeError('MJWarp contact capacity exceeded')
        result = [[] for _ in range(r.num_worlds)]
        if not count:
            return result
        # Reuse fixed buffers; contact_force ignores indices beyond nacon.
        if not hasattr(self, '_contact_ids'):
            self._contact_ids = r.wp.array(np.arange(d.naconmax, dtype=np.int32), dtype=r.wp.int32, device=r.device)
            self._contact_forces = r.wp.zeros(d.naconmax, dtype=r.wp.spatial_vector, device=r.device)
        with r.wp.ScopedDevice(r.device):
            r.mjw.contact_force(r.model, d, self._contact_ids, True, self._contact_forces)
        forces = self._contact_forces.numpy()[:count, :3] * r.timestep
        kinds = d.contact.type.numpy()[:count]
        worlds = d.contact.worldid.numpy()[:count]
        geoms = d.contact.geom.numpy()[:count]
        positions = d.contact.pos.numpy()[:count]
        exported = int(ContactType.CONSTRAINT | ContactType.PASSIVE)
        selected = np.flatnonzero(((kinds & exported) != 0) & (np.linalg.norm(forces[:, :2], axis=1) >= 3))
        model = r.cpu_model
        for c in selected:
            world = int(worlds[c])
            if not 0 <= world < r.num_worlds:
                raise RuntimeError(f'Invalid contact world {world}')
            for geom in geoms[c]:
                if not 0 <= int(geom) < model.ngeom:
                    raise RuntimeError(f'Invalid contact geometry {geom}')
                body = int(model.geom_bodyid[geom])
                name = self.mj.mj_id2name(model, self.mj.mjtObj.mjOBJ_BODY, body)
                if name and name.startswith('b'):
                    result[world].append(dict(body=int(name[1:]), pos=positions[c].tolist(), impulse=float(np.linalg.norm(forces[c]))))
        return result

    def result(self, request, cpu, state, world):
        runtime = self.runtime; model = runtime.cpu_model; np = self.np
        bodies = []
        impacts = self.contacts[world]
        for b in request['bodies']:
            i = self.body_ids[f"b{b['id']}"]
            if b['dynamic']:
                joint = model.body_jntadr[i]; q = model.jnt_qposadr[joint]; v = model.jnt_dofadr[joint]
                pos = state['qpos'][world, q:q+3]; quat = state['qpos'][world, q+3:q+7]
                velocity = state['qvel'][world, v:v+3]; angular = self.rotate(state['qvel'][world, v+3:v+6], quat)
            else:
                pos = b['next']['pos']; quat = b['next']['quat']; velocity = b['velocity']; angular = b['angular']
            bodies.append(dict(id=b['id'], pos=list(map(float, pos)), quat=list(map(float, quat)), velocity=list(map(float, velocity)), angular=list(map(float, angular))))
        return {'op': 'external_physics_result', 'bodies': bodies, 'impacts': impacts}

    def close(self):
        self.geometry_cache.clear()
        self.collider_cache.clear()
        if self.runtime:
            self.runtime.close()
            self.runtime = None
