# FalkorDB feedback from the inherit-brain build

Tested on `falkordb/falkordb:v4.22.1` and `falkordb/falkordb:6.0.2` with the Python client `falkordb==1.8.0`.

## 1. Version difference: GRAPH.UDF LIST reply shape depends on the version (RESP3 map vs array)

A client built with `FalkorDB.from_url(...)` gets a map per library from 4.22.1 and an array from 6.0.2.
A client built with `FalkorDB(host=..., port=...)` (RESP2) gets an array from both.

```python
from falkordb import FalkorDB
for port in (6379, 6380):   # 6379 = 4.22.1, 6380 = 6.0.2
    db = FalkorDB.from_url(f"redis://localhost:{port}")
    db.udf_load("demo", "function f(){return 1;}\nfalkor.register('f', f);")
    print(port, type(db.udf_list("demo", with_code=True)[0]).__name__)
    db.connection.execute_command("GRAPH.UDF", "DELETE", "demo")
# 6379 dict
# 6380 list
```

Effect: code that reads the library source back (to check that a loaded UDF matches the file) must accept both shapes.

## 2. ACL: read-only key selectors do not work for graph commands (both versions)

`COMMAND GETKEYSANDFLAGS GRAPH.RO_QUERY g RETURN 1` reports the key as `RW access update`, so a `%R~herb:p` rule never
matches and the user gets "No permissions to access a key" on a graph it should read. We use
`(~herb:p +GRAPH.RO_QUERY)` instead. GRAPH.RO_QUERY and GRAPH.COPY (source key) should declare a read flag.

```
ACL SETUSER u reset on >pw +GRAPH.RO_QUERY %R~herb:p
# as u:  GRAPH.RO_QUERY herb:p "RETURN 1"   -> NOPERM No permissions to access a key
```

## 3. Vector index accepts a wrong-dimension vector (both versions)

```
GRAPH.QUERY g "CREATE VECTOR INDEX FOR (m:Message) ON (m.embedding) OPTIONS {dimension:1024, similarityFunction:'cosine'}"
GRAPH.QUERY g "CREATE (:Message {id:1, embedding: vecf32([1.0, 2.0])})"    -> succeeds
```

The node is created and never errors. We validate the length in the loader.

## 4. Mandatory constraint is checked per clause, not per query (both versions)

`UNWIND $rows AS r CREATE (n:Bug) SET n = r.props` fails with "mandatory constraint violation" even though `r.props`
carries the mandatory property, because the node exists without it after `CREATE`. Putting the property in the
`CREATE` pattern (`CREATE (n:Bug {status: r.props.status}) SET n = r.props`) works. Worth a line in the constraint docs.

## 5. A restricted user needs +INFO to use the Python client

The client calls `INFO` when it connects, so a build user restricted to GRAPH commands fails with
"no permissions to run the 'info' command" until `+INFO` is granted.
