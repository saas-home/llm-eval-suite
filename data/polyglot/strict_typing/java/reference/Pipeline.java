import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;

public final class Pipeline<T> {
    private final List<T> data;

    public Pipeline(List<T> data) {
        this.data = List.copyOf(data);
    }

    public <U> Pipeline<U> map(Function<? super T, ? extends U> mapper) {
        return new Pipeline<U>(data.stream().<U>map(mapper).toList());
    }

    public Pipeline<T> filter(Predicate<? super T> predicate) {
        return new Pipeline<>(data.stream().filter(predicate).toList());
    }

    public List<T> execute() {
        return data;
    }
}
