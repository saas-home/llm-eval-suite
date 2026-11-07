import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;

public final class Pipeline<T> {
    private final List data;

    public Pipeline(List<T> data) {
        this.data = new ArrayList(data);
    }

    @SuppressWarnings("unchecked")
    public <U> Pipeline<U> map(Function<? super T, ? extends U> mapper) {
        List out = new ArrayList();
        for (Object o : data) out.add(mapper.apply((T) o));
        return new Pipeline<U>(out);
    }

    @SuppressWarnings("unchecked")
    public Pipeline<T> filter(Predicate<? super T> predicate) {
        List<T> out = new ArrayList<>();
        for (Object o : data) if (predicate.test((T) o)) out.add((T) o);
        return new Pipeline<>(out);
    }

    @SuppressWarnings("unchecked")
    public List<T> execute() {
        return new ArrayList<T>(data);
    }
}
