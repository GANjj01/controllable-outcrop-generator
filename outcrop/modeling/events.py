from dataclasses import dataclass
from .schema import thaw,freeze
from .query import StageSnapshot
from .errors import invalid


@dataclass(frozen=True)
class ObjectVersion:
    object_id: str
    stage_id: str


@dataclass(frozen=True)
class OccupancyBranch:
    source_region: object
    source_snapshot_id: str
    side_constraint: object
    map_chain: tuple
    body_id: str
    lineage: tuple


@dataclass(frozen=True)
class ExecutionPlan:
    events: tuple
    dependencies: dict


def compile_events(numeric):
    d=thaw(numeric.data);events={s["id"]:s for s in d.get("events",[])};bodies={s["id"] for s in d["bodies"]};geometry={s["id"] for s in d["geometry"]};deps={oid:set(s.get("after",[])) for oid,s in events.items()}
    stage_producer={s.get("output_stage",s["id"]):s["id"] for s in events.values()}
    if len(stage_producer)!=len(events) or "initial" in stage_producer:invalid("events","unique non-initial output stages required")
    for oid,e in events.items():
        if e["kind"] not in {"fold","deformation","intrusion","fault","field_deformation","cuts","truncates","create_features"}:invalid(oid,"unknown event kind")
        if e.get("input_stage","initial")!="initial":
            stage=e["input_stage"]
            if stage not in stage_producer:invalid(oid,"unknown input stage")
            deps[oid].add(stage_producer[stage])
        for target in e.get("targets",[]):
            if target not in bodies:invalid(oid,f"unknown target body {target}")
        if e["kind"] in {"fold","deformation","field_deformation"} and e.get("map") not in geometry:invalid(oid,"map reference required")
    indegree={oid:len(v) for oid,v in deps.items()};done=[];ready=sorted(oid for oid,n in indegree.items() if n==0)
    while ready:
        oid=ready.pop(0);done.append(oid)
        for child in sorted(events):
            if oid in deps[child]:
                indegree[child]-=1
                if indegree[child]==0:ready.append(child);ready.sort()
    if len(done)!=len(events):invalid("events","event dependency cycle")
    ancestors={oid:set(deps[oid]) for oid in done}
    for oid in done:
        for p in deps[oid]:ancestors[oid]|=ancestors[p]
    def writes(e):return set(e.get("targets",[]))|({e["body"]} if "body" in e else set())
    for i,a in enumerate(done):
        for b in done[i+1:]:
            if writes(events[a])&writes(events[b]) and a not in ancestors[b]:invalid("events",f"explicit order required for noncommuting events {a}/{b}")
    return ExecutionPlan(tuple(freeze(events[x]) for x in done),{k:tuple(sorted(v)) for k,v in deps.items()})
