static string Render(IEnumerable<(string Id, decimal Total)> rows) => "[" + string.Join(",", rows.Select(r => $"{r.Id}:{r.Total}")) + "]";

var specs = new List<(string? Id, int Qty, decimal Price, decimal? Discount)?[]?>
{
    null,
    new (string?, int, decimal, decimal?)?[] { },
    new (string?, int, decimal, decimal?)?[] { null },
    new (string?, int, decimal, decimal?)?[] { ("a", 2, 3m, null) },
    new (string?, int, decimal, decimal?)?[] { (null, 2, 3m, null), ("", 1, 1m, null) },
    new (string?, int, decimal, decimal?)?[] { ("q0", 0, 5m, null), ("qn", -1, 5m, null) },
    new (string?, int, decimal, decimal?)?[] { ("pn", 2, -1m, null), ("p0", 2, 0m, null) },
    new (string?, int, decimal, decimal?)?[] { ("d0", 4, 2.5m, 0m), ("d25", 4, 2.5m, 0.25m) },
    new (string?, int, decimal, decimal?)?[] { ("d1", 3, 7m, 1m), ("d15", 3, 7m, 1.5m), ("dn", 3, 7m, -0.2m) },
    new (string?, int, decimal, decimal?)?[] { ("half-even", 1, 0.125m, null), ("half-even-2", 1, 2.345m, null), ("r3", 7, 1.115m, 0.1m) },
};

foreach (var spec in specs)
{
    var legacyInput = spec?.Select(s => s is { } v ? new Legacy.Record(v.Id, v.Qty, v.Price, v.Discount) : null).ToList();
    var refactoredInput = spec?.Select(s => s is { } v ? new __NS__.Record(v.Id, v.Qty, v.Price, v.Discount) : null).ToList();
    var before = Render(Legacy.OrderProcessor.ProcessOrderData(legacyInput).Select(o => (o.Id, o.Total)));
    var after = Render(__NS__.OrderProcessor.ProcessOrderData(refactoredInput).Select(o => (o.Id, o.Total)));
    if (before != after) throw new Exception($"behavior changed: legacy {before} vs refactored {after}");
}
Console.WriteLine("POLYGLOT_OK " + Environment.GetEnvironmentVariable("EVAL_RUN_ID"));
