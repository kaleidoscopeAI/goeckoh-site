    189 +              padding: const EdgeInsets.only(bottom: 20.0),
    190 +              child: Column(
    191 +                mainAxisSize: MainAxisSize.min,
    192 +                crossAxisAlignment: CrossAxisAlignment.end,
    193 +                children: [
    194 +                  _Badge(label: "GCL", value: (_mirror["gcl"] ?? "").toString()),
    195 +                  const SizedBox(height: 6),
    196 +                  _Badge(label: "Mode", value: (_mirror["mode"] ?? "").toString()),
    197 +                  const SizedBox(height: 6),
    198 +                  _Badge(label: "Latency", value: (_mirror["latency_ms"] ?? "?").toString()),
    199 +                ],
    200 +              ),
    201 +            )
    202 +          : null,
    203 +    );
    204 +  }
    205 +}
    206 +
    207 +class _Badge extends StatelessWidget {
    208 +  final String label;
    209 +  final String value;
    210 +  const _Badge({required this.label, required this.value});
    211 +  @override
    212 +  Widget build(BuildContext context) {
    213 +    return Container(
    214 +      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
    215 +      decoration: BoxDecoration(
    216 +        color: Colors.black.withOpacity(0.6),
    217 +        borderRadius: BorderRadius.circular(12),
    218 +        border: Border.all(color: Colors.cyanAccent.withOpacity(0.4)),
    219 +      ),
    220 +      child: Text("$label: $value",
    221 +          style: const TextStyle(fontSize: 12, color: Colors.cyanAccent)),
    222      );

