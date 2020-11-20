import org.junit.Test
import kotlin.test.assertTrue

class GraphsTest {
    @Test
    fun isomorphism_simple() {
        val g1 = Matrix.fromString("{{0,1,0},{1,0,1},{0,1,1}}")
        val g2 = Matrix.fromString("{{0,1,1},{1,1,0},{1,0,0}}")
        assertTrue { Matrix.arePseudoIsomorphic(g1, g2) }
    }

    @Test
    fun isomorphism_failure_notGraph() {
        var failure = false
        try {
            val g1 = Matrix.fromString("{{0,2,0},{1,0,1},{0,1,1}}")
            val g2 = Matrix.fromString("{{0,1,1},{1,1,0},{1,0,0}}")
            val res = Matrix.arePseudoIsomorphic(g1, g2)
        } catch (e: Exception) {
            failure = true
        }
        assertTrue { failure }
    }

    @Test
    fun isomorphism_failure_notIsomorphic() {
        val g1 =
            Matrix.fromString("{{0,1,1,0,0,0},{1,0,0,1,0,0},{1,0,0,0,1,0},{0,1,0,0,0,1},{0,0,1,0,1,0},{0,0,0,1,0,0}}")
        val g2 =
            Matrix.fromString("{{1,1,0,0,0,0},{1,0,0,0,0,0},{0,0,0,1,1,0},{0,0,1,0,0,1},{0,0,1,0,0,1},{0,0,0,1,1,0}}")
        assertTrue { !Matrix.arePseudoIsomorphic(g1, g2) }
    }


}