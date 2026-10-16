// herb_v1.statusAt(events, date): status after the last event with ts <= date.
// events: list of {type, ts}; type in REPORTED, ASSIGNED, RESOLVED, DISMISSED, REOPENED.
function statusAt(events, date) {
    var best = null;
    var status = null;
    for (var i = 0; i < events.length; i++) {
        var e = events[i];
        if (e.ts <= date && (best === null || e.ts >= best)) {
            best = e.ts;
            if (e.type === 'RESOLVED') status = 'resolved';
            else if (e.type === 'DISMISSED') status = 'dismissed';
            else if (e.type === 'REPORTED' || e.type === 'ASSIGNED' || e.type === 'REOPENED') status = 'active';
        }
    }
    return status;
}
falkor.register('statusAt', statusAt);
