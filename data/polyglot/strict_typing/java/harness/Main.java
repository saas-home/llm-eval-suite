import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;

public class Main {
    static void check(boolean cond, String msg) { if (!cond) throw new AssertionError(msg); }

    public static void main(String[] args) {
        List<Integer> src = new ArrayList<>(List.of(1, 2, 3, 4, 5));
        Pipeline<Integer> p = new Pipeline<>(src);
        List<String> out = p.filter(n -> n % 2 == 1).map(n -> "#" + n).execute();
        check(out.equals(List.of("#1", "#3", "#5")), "filter/map/execute result wrong: " + out);
        src.set(0, 99);
        check(p.execute().get(0) == 1, "the constructor must copy its input (the pipeline is immutable)");
        List<Integer> got = p.execute();
        try {
            got.set(1, 77);
        } catch (UnsupportedOperationException expected) {
            // an unmodifiable view is also acceptable
        }
        check(p.execute().equals(List.of(1, 2, 3, 4, 5)), "execute() must not expose internal state");
        Predicate<Object> nonNull = Objects::nonNull;
        Function<Object, String> describe = Object::toString;
        List<String> viaWildcards = p.filter(nonNull).map(describe).execute();
        check(viaWildcards.size() == 5, "map/filter must accept ? super T wildcards");
        System.out.println("POLYGLOT_OK " + System.getenv("EVAL_RUN_ID"));
    }
}
