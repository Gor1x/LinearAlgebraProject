import ch.obermuhlner.math.big.BigDecimalMath
import java.math.BigDecimal
import java.math.MathContext
import java.math.RoundingMode
import java.util.*
import kotlin.math.abs
import kotlin.math.min

const val EPS = 1e-9
const val DEFAULT_PRECISION = 25
val DEFAULT_MATH_CONTEXT = MathContext(DEFAULT_PRECISION, RoundingMode.HALF_UP)

data class Circle(val middle: BigDecimal, val radius: BigDecimal) {
    override fun toString(): String {
        return "Circle(middle=${middle.setScale(3, RoundingMode.HALF_UP)}, radius=${
            radius.setScale(3, RoundingMode.HALF_UP)
        })"
    }
}

open class Matrix(private val height: Int, val width: Int = height) {
    val size: Int
        get() = if (height == width) height else throw IllegalStateException("Matrix is not squared")

    private var mt: Array<Array<BigDecimal>> = Array(height) { Array(width) { BigDecimal(0) } }

    constructor(a: Array<Array<BigDecimal>>) : this(a.size, a[0].size) {
        mt = a
        val sz = mt[0].size
        if (mt.any { it.size != sz })
            throw IllegalArgumentException("Wrong format matrix given")
    }

    constructor(v: Vector) : this(v.size, 1) {
        mt = Array(v.size) { line -> arrayOf(v[line]) }
    }

    operator fun times(v: Vector): Vector {
        return Vector(Array(mt.size) {
            mt[it].foldIndexed(BigDecimal(0)) { i, acc, d -> acc + d * v[i] }
                .setScale(DEFAULT_PRECISION, RoundingMode.HALF_EVEN)
        })
    }

    operator fun get(i: Int): Array<BigDecimal> {
        return mt[i]
    }

    operator fun times(k: BigDecimal): Matrix {
        return Matrix(mt.map { arr ->
            arr.map { (it * k).setScale(DEFAULT_PRECISION, RoundingMode.HALF_EVEN) }.toTypedArray()
        }.toTypedArray())
    }

    operator fun times(m2: Matrix): Matrix {
        if (width != m2.height)
            throw java.lang.IllegalArgumentException("Wrong matrix size multiplication")
        val arr = Array(height) { i ->
            Array(m2.width) { j ->
                var result = BigDecimal.ZERO
                for (k in 0 until width)
                    result += mt[i][k] * m2.mt[k][j]
                result.setScale(DEFAULT_PRECISION, RoundingMode.HALF_EVEN)
            }
        }
        return Matrix(arr)
    }

    operator fun plus(matrix: Matrix): Matrix {
        if (height != matrix.height || width != matrix.width)
            throw java.lang.IllegalArgumentException("Matrix sizes doesn't match")
        return Matrix(Array(height) { i ->
            Array(width) { j ->
                (mt[i][j] + matrix[i][j]).setScale(DEFAULT_PRECISION, RoundingMode.HALF_EVEN)
            }
        })
    }

    operator fun minus(matrix: Matrix): Matrix {
        if (height != matrix.height || width != matrix.width)
            throw java.lang.IllegalArgumentException("Matrix sizes doesn't match")
        return Matrix(Array(height) { i ->
            Array(width) { j ->
                (mt[i][j] - matrix[i][j]).setScale(DEFAULT_PRECISION, RoundingMode.HALF_EVEN)
            }
        })
    }

    operator fun minusAssign(newMatrix: Matrix) {
        for (i in 0 until height)
            for (j in 0 until width)
                mt[i][j] = (mt[i][j] - newMatrix[i][j]).setScale(DEFAULT_PRECISION, RoundingMode.HALF_EVEN)
    }


    private fun dimensions(): Pair<Int, Int> {
        return mt.size to mt[0].size
    }

    fun isZero(): Boolean {
        return mt.all { arr ->
            arr.all { it < BigDecimal(1e-9) }
        }
    }

    operator fun unaryMinus(): Matrix {
        return this * BigDecimal(-1)
    }

    override fun toString(): String {
        return mt.joinToString(prefix = "[", postfix = "]\n", separator = "\n") {
            it.joinToString(
                prefix = "{",
                postfix = "}",
                separator = ","
            ) {
                it.setScale(3, RoundingMode.HALF_UP).toString()
            }
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Matrix

        if (height != other.height) return false
        if (width != other.width) return false

        for (i in 0 until height)
            for (j in 0 until width)
                if (mt[i][j].setScale(DEFAULT_PRECISION / 2, RoundingMode.HALF_UP) != other.mt[i][j].setScale(
                        DEFAULT_PRECISION / 2,
                        RoundingMode.HALF_UP
                    )
                ) return false

        return true
    }

    override fun hashCode(): Int {
        var result = height
        result = 31 * result + width
        result = 31 * result + mt.contentDeepHashCode()
        return result
    }

    fun transposed(): Matrix {
        return Matrix(Array(width) { j ->
            Array(height) { i ->
                mt[i][j].setScale(DEFAULT_PRECISION, RoundingMode.HALF_EVEN)
            }
        })
    }

    fun equalsWithAccuracy(other: Matrix, eps: Double): Boolean {
        for (i in 0 until height)
            for (j in 0 until width)
                if ((mt[i][j] - other.mt[i][j]).abs() > eps.toBigDecimal())
                    return false
        return true
    }

    companion object {
        fun read(n: Int, s: String = ""): Matrix {
            return Matrix(Array(n) {
                readLine()!!.split(' ').map { it.toDouble().toBigDecimal() }.toTypedArray()
            })
        }

        fun getIdentity(size: Int): Matrix {
            return Matrix(Array(size) { i ->
                Array(size) { j ->
                    if (i == j) BigDecimal.ONE else BigDecimal.ZERO
                }
            })
        }

        fun gershgorinCircles(m: Matrix): Array<Circle> {
            return Array(m.height) { ind ->
                var sum = BigDecimal.ZERO
                for (i in 0 until m.width)
                    sum += if (i == ind)
                        BigDecimal.ZERO
                    else
                        m[ind][i].abs()
                Circle(m[ind][ind], sum)
            }
        }

        fun diagToVector(m: Matrix): Vector {
            val size = min(m.width, m.height)
            return Vector(Array(size) { i -> m[i][i] })
        }

        fun getDiagPartOfMatrix(m: Matrix): Matrix {
            return Matrix(m.mt.mapIndexed { j, arr ->
                arr.mapIndexed { i, elem ->
                    if (i == j) elem else BigDecimal(0)
                }.toTypedArray()
            }.toTypedArray())
        }

        fun getUpperFrom(m: Matrix): Matrix {
            return Matrix(m.mt.mapIndexed { j, arr ->
                arr.mapIndexed { i, elem ->
                    if (j > i) elem else BigDecimal(0)
                }.toTypedArray()
            }.toTypedArray())
        }

        fun multiplyOnHouseholderMatrix(m: Matrix, v: Vector) {
            val vector = Matrix(v)
            val newMatrix = (vector * BigDecimal(2)) * (vector.transposed() * m)
            m -= newMatrix
        }

        fun multiplyOnHouseholderMatrixFromRight(m: Matrix, v: Vector) {
            val vector = Matrix(v)
            val newMatrix = m * (vector * BigDecimal(2)) * vector.transposed()
            m -= newMatrix
        }

        fun getQRdecompositionWithHouseholderMethod(m: Matrix): Pair<Matrix, Matrix> {
            val matrixR = m.copy()
            val size = matrixR.width
            val multiplications = mutableListOf<Vector>()
            for (i in 0 until size) {
                val uVec = Vector(Array(size) { ind ->
                    if (ind < i) BigDecimal.ZERO else matrixR[ind][i]
                })
                val u = uVec.normalized()
                val v = (u - Vector.unitVector(size, i)).normalized()
                multiplyOnHouseholderMatrix(matrixR, v)
                multiplications.add(v)
            }
            val matrixQ = getIdentity(size)
            multiplications.reversed().forEach { vec -> multiplyOnHouseholderMatrix(matrixQ, vec) }
            return matrixQ to matrixR
        }

        fun getTridiagonal(m: Matrix, needQ: Boolean = true): Pair<Matrix, Matrix> {
            if (m != m.transposed())
                throw IllegalArgumentException("Matrix must be symmetrical")
            val matrixR = m.copy()
            val size = matrixR.width
            val transposedQ = getIdentity(size)
            for (i in 0 until size - 1) {
                val uVec = Vector(Array(size) { ind ->
                    if (ind <= i) BigDecimal.ZERO else matrixR[ind][i]
                })
                val u = uVec.normalized()
                val v = (u - Vector.unitVector(size, i + 1)).normalized()
                for (j in 0..i)
                    v[j] = BigDecimal.ZERO
                multiplyOnHouseholderMatrixFromRight(matrixR, v)
                multiplyOnHouseholderMatrix(matrixR, v)
                if (needQ)
                    multiplyOnHouseholderMatrix(transposedQ, v)
            }
            return matrixR to transposedQ.transposed()
        }

        fun getLowerFrom(m: Matrix): Matrix {
            return Matrix(m.mt.mapIndexed { j, arr ->
                arr.mapIndexed { i, elem ->
                    if (j < i) elem else BigDecimal(0)
                }.toTypedArray()
            }.toTypedArray())
        }

        fun multiplyOnGivensMatrix(m: Matrix, i: Int, j: Int, phi: BigDecimal) {
            val c = BigDecimalMath.cos(phi, DEFAULT_MATH_CONTEXT)
            val s = BigDecimalMath.sin(phi, DEFAULT_MATH_CONTEXT)
            val ith = Vector(m.mt[i])
            val jth = Vector(m.mt[j])

            val newIth = (ith * c + jth * s).toArray()
            val newJth = (-ith * s + jth * c).toArray()
            m.mt[i] = newIth
            m.mt[j] = newJth
        }

        fun getEigenvectors(m: Matrix, givenEps: Double = 1e-3, needQ: Boolean = true): Pair<Vector, Matrix> {
            var matrix = m.copy()
            val eps = givenEps.toBigDecimal()
            var answerQ = getIdentity(m.size)
            val isTridiag = isTridiagonal(m)
            while (true) {
                for (i in 0..15) {
                    val ans = getQRdecompositionWithGivensRotation(matrix, isTridiag = isTridiag)
                    val Q = ans.first
                    val R = ans.second
                    if (!isTridiag || needQ)
                        answerQ *= Q //Here is a problem because of O(n^3)
                    matrix = R * Q
                }
                if (gershgorinCircles(matrix).all { it.radius < eps }) {
                    return diagToVector(matrix) to answerQ
                }
            }
        }

        fun getEigenvectorsFastTridiagonal(
            m: Matrix,
            givenEps: Double = 1e-3,
            needQ: Boolean = true
        ): Pair<Vector, Matrix> {
            var matrix = m.copy()
            val eps = givenEps.toBigDecimal()
            var answerQ = getIdentity(m.size)
            val isTridiag = isTridiagonal(m)

            for (i in 0..25) {
                val ans = getQRdecompositionWithGivensRotation(matrix, isTridiag = isTridiag)
                val Q = ans.first
                val R = ans.second
                if (needQ)
                    answerQ *= Q
                matrix = R * Q
            }

            var diag = m.size - 1
            var sMatrix = getIdentity(m.size) * matrix[diag][diag]
            while (true) {
                for (i in 0..5) {
                    val ans = getQRdecompositionWithGivensRotation(matrix - sMatrix, isTridiag = isTridiag)
                    val Q = ans.first
                    val R = ans.second
                    if (needQ)
                        answerQ *= Q
                    matrix = R * Q + sMatrix
                }

                if (diag != 0 && matrix[diag][diag - 1].abs() < eps && matrix[diag - 1][diag].abs() < eps) {
                    diag--
                    sMatrix = getIdentity(m.size) * matrix[diag][diag]
                }
                if (diag == 0)
                    break

            }
            return diagToVector(matrix) to answerQ
        }

        //Задача 4 и Задача 3 как подзадача данной
        fun getQRdecompositionWithGivensRotation(matrix: Matrix, isTridiag: Boolean = false): Pair<Matrix, Matrix> {
            fun removeElement(
                m: Matrix,
                i: Int,
                j: Int,
                removes: MutableList<Triple<Int, Int, BigDecimal>>,
                column: Int = j
            ) {
                val x_i = m[i][column]
                val x_j = m[j][column]
                val bot = (x_i * x_i + x_j * x_j).sqrt(DEFAULT_MATH_CONTEXT)
                val y = -x_i.divide(bot, DEFAULT_MATH_CONTEXT)
                val x = x_j.divide(bot, DEFAULT_MATH_CONTEXT)
                val angle = BigDecimalMath.atan2(y, x, DEFAULT_MATH_CONTEXT)
                removes.add(Triple(i, j, angle))
                multiplyOnGivensMatrix(m, i, j, angle)
            }

            val matrixR = matrix.copy()
            val removes: MutableList<Triple<Int, Int, BigDecimal>> = mutableListOf()
            for (column in 0 until matrixR.width) {
                val maxHeightTo = if (isTridiag)
                    min(column + 3, matrixR.height)
                else
                    matrixR.height

                var curIndex = column
                //Try to find first zero in this column
                while (curIndex < maxHeightTo && matrixR[curIndex][column].abs() < EPS.toBigDecimal())
                    curIndex++
                //Maybe it is all zero so we need to exit
                if (curIndex >= maxHeightTo)
                    continue
                if (curIndex != column) {
                    //Move element which is not zero to a diag
                    val pidiv2 = BigDecimalMath.pi(DEFAULT_MATH_CONTEXT) / BigDecimal(2)
                    multiplyOnGivensMatrix(matrixR, column, curIndex, pidiv2)
                    removes.add(Triple(column, curIndex, pidiv2))
                }
                curIndex++
                while (curIndex < maxHeightTo) {
                    if (matrixR[curIndex][column].setScale(
                            DEFAULT_PRECISION / 2,
                            RoundingMode.HALF_UP
                        ) != BigDecimal.ZERO
                    )
                        removeElement(matrixR, i = curIndex, j = column, removes = removes)
                    curIndex++
                }
            }
            val matrixQ = getIdentity(matrixR.width)
            removes.reversed().forEach { (i, j, phi) ->
                multiplyOnGivensMatrix(matrixQ, i, j, -phi)
            }
            return matrixQ to matrixR
        }

        private fun isTridiagonal(matrix: Matrix): Boolean {
            for (i in 0 until matrix.height)
                for (j in 0 until matrix.width)
                    if (abs(i - j) >= 2 && matrix[i][j].abs() > EPS.toBigDecimal())
                        return false
            return true
        }

        fun buildFromDiag(v: Vector): Matrix {
            return Matrix(Array(v.size) { i ->
                Array(v.size) { j ->
                    if (i == j)
                        v[i]
                    else
                        BigDecimal.ZERO
                }
            })
        }

        fun fromString(s: String): Matrix {
            val arr = s.filter { it != ' ' && it != '\n' }.drop(1).dropLast(1).split("}").map { line ->
                line.filter { it != '{' && it != '}' }.split(",").filter { it.isNotEmpty() }.map { it.toBigDecimal() }
                    .toTypedArray()
            }.filter { it.isNotEmpty() }.toTypedArray()
            return Matrix(arr)
        }

        fun arePseudoIsomorphic(g1: Matrix, g2: Matrix): Boolean {
            fun isGraph(m: Matrix) = m.mt.all { line ->
                line.all { it == BigDecimal.ONE || it == BigDecimal.ZERO }
            }
            if (!isGraph(g1))
                throw IllegalArgumentException("First argument is not a graph")
            if (!isGraph(g2))
                throw IllegalArgumentException("First argument is not a graph")

            fun degrees(g: Matrix): List<Double> = g.mt.map { line ->
                line.fold(0.0) { acc, it ->
                    acc + it.toDouble()
                }
            }

            val degreesG1 = degrees(g1)
            val degreesG2 = degrees(g2)

            if (degreesG1.sorted() != degreesG2.sorted())
                return false

            val (tridiagG1, _) = getTridiagonal(g1, false)
            val (tridiagG2, _) = getTridiagonal(g2, false)

            val (vecG1, _) = getEigenvectorsFastTridiagonal(tridiagG1, 1e-2, false)
            val (vecG2, _) = getEigenvectorsFastTridiagonal(tridiagG2, 1e-2, false)
            if ((vecG1 - vecG2).toArray().any { it > 1e-2.toBigDecimal() })
                return false

            return true
        }

        fun findOptimalAlpha(value: Int, isP: Boolean): BigDecimal {
            lateinit var m: Matrix
            if (isP) {
                fun binpow(x: Int, y: Int, mod: Int): Int {
                    if (y <= 0)
                        return 1
                    if (y % 2 == 1)
                        return binpow(x, y - 1, mod) * x % mod
                    return ({e : Int -> e * e }(binpow(x, y / 2, mod))) % mod
                }
                fun opposite(x:Int, p:Int) = if (x == 0) 0 else binpow(x, p - 2, p)

                val p = value
                m = Matrix(p)
                for (x in 1 until p) {
                    val prev = if (x == 0) p - 1 else x - 1
                    val next = if (x == p - 1) 0 else x + 1
                    val opp = opposite(x, p)
                    m[x][prev] += BigDecimal.ONE
                    m[x][next] += BigDecimal.ONE
                    m[x][opp] += BigDecimal.ONE
                }
            } else {
                val n = value
                fun toNum(x: Int, y: Int) = ((x % n + n) % n) * n + ((y % n + n) % n)
                m = Matrix(n * n)
                for (x in 0 until n)
                    for (y in 0 until n)
                        for (sgn in listOf(-1, 1)) {
                            m[toNum(x, y)][toNum(x + 2 * y * sgn, y)] += BigDecimal.ONE

                            m[toNum(x, y)][toNum(x + (2 * y + 1) * sgn, y)] += BigDecimal.ONE

                            m[toNum(x, y)][toNum(x, y + 2 * x * sgn)] += BigDecimal.ONE

                            m[toNum(x, y)][toNum(x, y + (2 * x + 1) * sgn)] += BigDecimal.ONE
                        }

            }
            val d = m.mt[0].reduce { acc, val2 -> acc + val2 }
            for (i in 0 until m.height)
                if (m[i].reduce { acc, v -> acc + v } != d)
                    throw IllegalArgumentException("Graph is not regular")
            val eigenvector = getEigenvectorsFastTridiagonal(getTridiagonal(m, false).first, 1e-5, false).first
            return (eigenvector[1].abs().max(eigenvector[eigenvector.size - 1].abs())).divide(
                d,
                DEFAULT_MATH_CONTEXT
            )
        }
    }

    private fun copy(): Matrix {
        return Matrix(mt.map { vec ->
            vec.map { it * BigDecimal.ONE }.toTypedArray().copyOf()
        }.toTypedArray().copyOf())

    }
}

class Vector(n: Int) {
    private var vec: Array<BigDecimal> = Array(n) { BigDecimal(0) }
    val size: Int
        get() = vec.size

    constructor(info: Array<BigDecimal>) : this(info.size) {
        vec = info
    }


    override fun equals(other: Any?): Boolean {
        return toString() == other.toString()
    }

    fun toArray(): Array<BigDecimal> {
        return vec.toList().toTypedArray()
    }

    fun norm(): BigDecimal {
        return vec.fold(BigDecimal(0)) { acc, d -> acc + d * d }.sqrt(DEFAULT_MATH_CONTEXT)
    }

    operator fun get(index: Int): BigDecimal {
        return vec[index]
    }

    operator fun plus(vector: Vector): Vector {
        if (size != vector.size) throw IllegalArgumentException("Different length of vectors")
        return Vector(vec.mapIndexed { i, d -> (d + vector.vec[i]).setScale(DEFAULT_PRECISION, RoundingMode.HALF_EVEN) }
            .toTypedArray())
    }

    operator fun minus(vector: Vector): Vector {
        return plus(vector * BigDecimal(-1))
    }

    operator fun times(k: BigDecimal): Vector {
        return Vector(vec.map { (it * k).setScale(DEFAULT_PRECISION, RoundingMode.HALF_EVEN) }.toTypedArray())
    }

    operator fun unaryMinus(): Vector {
        return this * BigDecimal(-1)
    }

    override fun toString(): String {
        return vec.joinToString(prefix = "[", postfix = "]", separator = ",") {
            it.setScale(3, RoundingMode.HALF_UP).toString()
        }
    }

    override fun hashCode(): Int {
        var result = size
        result = 31 * result + vec.contentHashCode()
        return result
    }

    fun normalized(): Vector {
        val len = if (norm().abs() < EPS.toBigDecimal())
            BigDecimal(1)
        else
            norm()
        return Vector(vec.map { it.divide(len, DEFAULT_MATH_CONTEXT) }.toTypedArray())
    }

    operator fun set(i: Int, value: BigDecimal) {
        vec[i] = value
    }

    companion object {
        fun read(n: Int): Vector {
            val arr =
                readLine()!!.split(' ').map { it.toDouble().toBigDecimal(DEFAULT_MATH_CONTEXT) }.toTypedArray()
            if (arr.size != n)
                throw IllegalArgumentException("Wrong vector length")
            return Vector(arr)
        }

        fun buildRandom(n: Int): Vector {
            val r = Random()
            return Vector(Array(n) { r.nextDouble().toBigDecimal(DEFAULT_MATH_CONTEXT) })
        }

        fun zero(n: Int): Vector {
            return Vector(Array(n) { BigDecimal.ZERO })
        }

        fun unitVector(length: Int, pos: Int): Vector {
            return Vector(Array(length) { ind ->
                if (ind == pos)
                    BigDecimal.ONE
                else
                    BigDecimal.ZERO
            })
        }

        fun buildOf(vararg elements: Int): Vector {
            val array = elements.map { it.toBigDecimal() }.toTypedArray()
            return Vector(array)
        }

        fun buildOf(vararg elements: Double): Vector {
            val array = elements.map { it.toBigDecimal() }.toTypedArray()
            return Vector(array)
        }
    }
}

//Задача 1
fun simpleIterationMethod(): Vector? {
    val n = readLine()!!.toInt()
    val a = Matrix.read(n)
    val b = Vector.read(n)
    val eps = readLine()!!.toBigDecimal()

    if (Matrix.gershgorinCircles(a).all { (c, e) -> (c - e).abs().max((c + e).abs()) >= BigDecimal.ONE }) {
        return null
    }

    var x = Vector.buildRandom(n)

    var timer = 0
    while (true) {
        val nx = a * x + b
        if ((nx - (a * nx) - b).norm() < eps)
            return x
        if (nx.norm() > x.norm() + BigDecimal.ONE) { //TODO() norm troubles
            timer++
            if (timer >= 20)
                return null
        }

        x = nx
    }
}

//Задача 2
fun gaussSeidelMethod(): Vector? {
    val n = readLine()!!.toInt()
    val a = Matrix.read(n)
    val b = Vector.read(n)
    val eps = readLine()!!.toBigDecimal()

    if (Matrix.getDiagPartOfMatrix(a).isZero()) {
        return null
    }
    val triangleUpper = Matrix.getUpperFrom(a)
    val triangleLower = Matrix.getLowerFrom(a) + Matrix.getDiagPartOfMatrix(a)

    var x = Vector.buildRandom(n)
    var badCounter = 0

    while (true) {
        if ((a * x - b).norm() < eps) {
            return x
        }
        val right = -triangleUpper * x + b
        val vecData = Array(n) { BigDecimal.ZERO }
        val nx = Vector(vecData.mapIndexed { line, _ ->
            (right[line] - triangleLower[line].foldIndexed(BigDecimal.ZERO, { i, sum, elem ->
                sum + (if (i < line) vecData[i] * elem
                else BigDecimal.ZERO)
            })) / triangleLower[line][line]
        }.toTypedArray())

        if (nx.norm() > x.norm() + BigDecimal.ONE) {
            badCounter++
            if (badCounter >= 20)
                return null
        }
        x = nx
    }
}


fun main(args: Array<String>) {
    println(Matrix.findOptimalAlpha(3, true))
}