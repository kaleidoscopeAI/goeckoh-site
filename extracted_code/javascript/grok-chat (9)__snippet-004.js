    81 -  const rawGcl = systemStatus.gcl_history || [
    81 +  const rawGcl = (systemStatus.gcl_history && systemStatus.gcl_history.l
        ength)
    82 +    ? systemStatus.gcl_history.map((v: number, i: number) => ({
    83 +        time: systemStatus.gcl_ts?.[i]
    84 +          ? new Date(systemStatus.gcl_ts[i] * 1000).toLocaleTimeString([
        ], { hour12: false, hour: '2-digit', minute: '2-digit' })
    85 +          : `${i}`,
    86 +        value: v,
    87 +      }))
    88 +    : [
    89      { time: '08:00', value: 0.72 },

