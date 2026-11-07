static void Check(bool cond, string msg) { if (!cond) throw new Exception(msg); }

var src = new List<int> { 1, 2, 3, 4, 5 };
var p = new Pipeline<int>(src);
IReadOnlyList<string> output = p.Filter(n => n % 2 == 1).Map(n => $"#{n}").Execute();
Check(output.SequenceEqual(new[] { "#1", "#3", "#5" }), "Filter/Map/Execute result wrong");
src[0] = 99;
Check(p.Execute()[0] == 1, "the constructor must copy its input (the pipeline is immutable)");
var got = p.Execute();
if (got is IList<int> list && !list.IsReadOnly) { list[1] = 77; }
Check(p.Execute().SequenceEqual(new[] { 1, 2, 3, 4, 5 }), "Execute() must not expose internal mutable state");
Console.WriteLine("POLYGLOT_OK " + Environment.GetEnvironmentVariable("EVAL_RUN_ID"));
