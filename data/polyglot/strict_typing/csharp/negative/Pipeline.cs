public sealed class Pipeline<T> {
    private readonly IReadOnlyList<T> _data;

    public Pipeline(IEnumerable<T> data) {
        _data = data.ToList().AsReadOnly();
    }

    public Pipeline<U> Map<U>(Func<T, U> mapper) {
        var boxed = _data.Select(x => (dynamic)mapper(x)!);
        return new Pipeline<U>(boxed.Select(b => (U)b));
    }

    public Pipeline<T> Filter(Func<T, bool> predicate) => new Pipeline<T>(_data.Where(predicate));

    public IReadOnlyList<T> Execute() => _data;
}
