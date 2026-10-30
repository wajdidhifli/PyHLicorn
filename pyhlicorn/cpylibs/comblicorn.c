#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <numpy/arrayobject.h>

// Org C Code 
void combLicorn(int * coactexp, int * ncoacts, int * corepexp , int * ncoreps, int * gexp, int * nsamples,  double * result , int * iact,int * irep){    
    int nsamp = (int)nsamples[0];
    int ncoact =(int)ncoacts[0];
    int ncorep = (int)ncoreps[0];
    int i =0;
    int j =0;
    int k=0;
    int  gexpred;
    int nonNormalizedMAE=0;
    for( i =0; i< ncoact;i++){
        for(  j =0; j< ncorep;j++){
            nonNormalizedMAE=0;
            for(  k =0; k <nsamp;k++){
                gexpred= coactexp[(i * nsamp)+k] - corepexp[(j * nsamp)+k];
                if (gexpred >1)  {
                    gexpred=1;
                } else if(gexpred < -1){
                    gexpred = -1;
                }
                nonNormalizedMAE += abs(gexp[k] -  gexpred);
            }
            result[(i*ncorep)+j]= (nonNormalizedMAE *1.0)/(nsamp *1.0);
            iact[(i*ncorep)+j]=i+1;
            irep[(i*ncorep)+j]=j+1;
        }
    }
}

// Wrapper for it to work cross-platform and for packaging

// 1 Input Declaration, verification, then conversion
// Boilerplates //
// 2 Declare method
// 3 Declare module as in file
// 4 Create module to be recognized

/* Reject unsafe buffers before handing raw pointers to the scoring loop. */
static int valid_buffer(PyObject *obj, int dtype, npy_intp size, int writable) {
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "combLicorn expects NumPy arrays");
        return 0;
    }
    PyArrayObject *arr = (PyArrayObject *)obj;
    if (PyArray_TYPE(arr) != dtype || !PyArray_ISCARRAY_RO(arr) ||
        !PyArray_ISNOTSWAPPED(arr) || PyArray_SIZE(arr) != size ||
        (writable && !PyArray_ISWRITEABLE(arr))) {
        PyErr_SetString(PyExc_ValueError, "Invalid combLicorn buffer dtype, layout, size, or writeability");
        return 0;
    }
    return 1;
}

static PyObject* py_combLicorn(PyObject* self, PyObject* args) {
    PyObject *coactexp_obj;
    int ncoacts;
    PyObject *corepexp_obj;
    int ncoreps;
    PyObject *gexp_obj;
    int nsamples;
    PyObject *result_obj, *iact_obj, *irep_obj;

    // O = Object , i = interger // see https://docs.python.org/3/c-api/arg.html#other-objects
    if (!PyArg_ParseTuple(args, "OiOiOiOOO", &coactexp_obj, &ncoacts, &corepexp_obj, &ncoreps, &gexp_obj, &nsamples, &result_obj, &iact_obj, &irep_obj)) {
        return NULL;
    }

    if (ncoacts < 1 || ncoreps < 1 || nsamples < 1 ||
        (npy_intp)ncoacts * ncoreps > INT_MAX ||
        (npy_intp)ncoacts * nsamples > INT_MAX ||
        (npy_intp)ncoreps * nsamples > INT_MAX || nsamples > INT_MAX / 2) {
        PyErr_SetString(PyExc_ValueError, "Invalid or excessive combLicorn dimensions");
        return NULL;
    }
    npy_intp pairs = (npy_intp)ncoacts * ncoreps;
    if (!valid_buffer(coactexp_obj, NPY_INT32, (npy_intp)ncoacts * nsamples, 0) ||
        !valid_buffer(corepexp_obj, NPY_INT32, (npy_intp)ncoreps * nsamples, 0) ||
        !valid_buffer(gexp_obj, NPY_INT32, nsamples, 0) ||
        !valid_buffer(result_obj, NPY_DOUBLE, pairs, 1) ||
        !valid_buffer(iact_obj, NPY_INT32, pairs, 1) ||
        !valid_buffer(irep_obj, NPY_INT32, pairs, 1)) return NULL;

    // Casting
    int *coactexp = (int*) PyArray_DATA((PyArrayObject*) coactexp_obj);
    int *corepexp = (int*) PyArray_DATA((PyArrayObject*) corepexp_obj);
    int *gexp = (int*) PyArray_DATA((PyArrayObject*) gexp_obj);
    double *result = (double*) PyArray_DATA((PyArrayObject*) result_obj);
    int *iact = (int*) PyArray_DATA((PyArrayObject*) iact_obj);
    int *irep = (int*) PyArray_DATA((PyArrayObject*) irep_obj);

    combLicorn(coactexp, &ncoacts, corepexp, &ncoreps, gexp, &nsamples, result, iact, irep);

    /* Py_BuildValue with O already increments each reference. */
    return Py_BuildValue("OOO", result_obj, iact_obj, irep_obj);
}


/* Membership buffers have one sorted, duplicate-free regulator list per model.
   Validate offsets before indexing raw pointers, including empty sentinel rows. */
static int valid_membership(PyObject *flat_obj, PyObject *offset_obj, int models) {
    if (!PyArray_Check(flat_obj)) {
        PyErr_SetString(PyExc_TypeError, "Regulator membership must be a NumPy array");
        return 0;
    }
    npy_intp size = PyArray_SIZE((PyArrayObject *)flat_obj);
    if (!valid_buffer(flat_obj, NPY_INT32, size, 0) ||
        !valid_buffer(offset_obj, NPY_INT64, (npy_intp)models + 1, 0)) return 0;
    const int *flat = (const int *)PyArray_DATA((PyArrayObject *)flat_obj);
    const npy_int64 *offset = (const npy_int64 *)PyArray_DATA((PyArrayObject *)offset_obj);
    if (offset[0] != 0 || offset[models] != size) {
        PyErr_SetString(PyExc_ValueError, "Membership offsets must span the flat buffer");
        return 0;
    }
    for (int i = 0; i < models; ++i) {
        if (offset[i] < 0 || offset[i+1] < offset[i] || offset[i+1] > size) {
            PyErr_SetString(PyExc_ValueError, "Invalid membership offsets");
            return 0;
        }
        for (npy_int64 j = offset[i]; j < offset[i+1]; ++j) {
            if (flat[j] < 0 || (j > offset[i] && flat[j] <= flat[j-1])) {
                PyErr_SetString(PyExc_ValueError, "Membership rows must be sorted, unique and nonnegative");
                return 0;
            }
        }
    }
    return 1;
}

/* Same scoring and row-major order as combLicorn. Invalid pairs are excluded
   before scoring, and valid rows are packed into the caller's output buffers. */
static PyObject *py_combLicornDisjoint(PyObject *self, PyObject *args) {
    PyObject *act_obj, *rep_obj, *gene_obj, *af_obj, *ao_obj, *rf_obj, *ro_obj;
    PyObject *result_obj, *ia_obj, *ir_obj;
    int nact, nrep, nsamples;
    if (!PyArg_ParseTuple(args, "OiOiOiOOOOOOO", &act_obj, &nact, &rep_obj, &nrep,
                          &gene_obj, &nsamples, &af_obj, &ao_obj, &rf_obj, &ro_obj,
                          &result_obj, &ia_obj, &ir_obj)) return NULL;
    if (nact < 1 || nrep < 1 || nsamples < 1 ||
        (npy_intp)nact * nrep > INT_MAX ||
        (npy_intp)nact * nsamples > INT_MAX ||
        (npy_intp)nrep * nsamples > INT_MAX || nsamples > INT_MAX / 2) {
        PyErr_SetString(PyExc_ValueError, "Invalid or excessive combLicorn dimensions");
        return NULL;
    }
    npy_intp pairs = (npy_intp)nact * nrep;
    if (!valid_buffer(act_obj, NPY_INT32, (npy_intp)nact * nsamples, 0) ||
        !valid_buffer(rep_obj, NPY_INT32, (npy_intp)nrep * nsamples, 0) ||
        !valid_buffer(gene_obj, NPY_INT32, nsamples, 0) ||
        !valid_buffer(result_obj, NPY_DOUBLE, pairs, 1) ||
        !valid_buffer(ia_obj, NPY_INT32, pairs, 1) ||
        !valid_buffer(ir_obj, NPY_INT32, pairs, 1) ||
        !valid_membership(af_obj, ao_obj, nact) ||
        !valid_membership(rf_obj, ro_obj, nrep)) return NULL;
    const int *act = (const int *)PyArray_DATA((PyArrayObject *)act_obj);
    const int *rep = (const int *)PyArray_DATA((PyArrayObject *)rep_obj);
    const int *gene = (const int *)PyArray_DATA((PyArrayObject *)gene_obj);
    const int *af = (const int *)PyArray_DATA((PyArrayObject *)af_obj);
    const int *rf = (const int *)PyArray_DATA((PyArrayObject *)rf_obj);
    const npy_int64 *ao = (const npy_int64 *)PyArray_DATA((PyArrayObject *)ao_obj);
    const npy_int64 *ro = (const npy_int64 *)PyArray_DATA((PyArrayObject *)ro_obj);
    double *result = (double *)PyArray_DATA((PyArrayObject *)result_obj);
    int *ia = (int *)PyArray_DATA((PyArrayObject *)ia_obj);
    int *ir = (int *)PyArray_DATA((PyArrayObject *)ir_obj);
    /* Restrict the scoring alphabet so integer subtraction/abs cannot overflow. */
    for (npy_intp i = 0; i < (npy_intp)nact * nsamples; ++i) {
        if (act[i] < -1 || act[i] > 1) {
            PyErr_SetString(PyExc_ValueError, "Activator states must be -1, 0, 1"); return NULL;
        }
    }
    for (npy_intp i = 0; i < (npy_intp)nrep * nsamples; ++i) {
        if (rep[i] != -1 && rep[i] != 0 && rep[i] != 2) {
            PyErr_SetString(PyExc_ValueError, "Repressor states must be -1, 0, 2"); return NULL;
        }
    }
    for (int i = 0; i < nsamples; ++i) {
        if (gene[i] < -1 || gene[i] > 1) {
            PyErr_SetString(PyExc_ValueError, "Target states must be -1, 0, 1"); return NULL;
        }
    }
    Py_ssize_t count = 0;
    for (int i = 0; i < nact; ++i) {
        for (int j = 0; j < nrep; ++j) {
            npy_int64 a = ao[i], r = ro[j];
            if (a == ao[i+1] && r == ro[j+1]) continue;
            int overlap = 0;
            while (a < ao[i+1] && r < ro[j+1]) {
                if (af[a] == rf[r]) { overlap = 1; break; }
                if (af[a] < rf[r]) ++a; else ++r;
            }
            if (overlap) continue;
            int error = 0;
            for (int k = 0; k < nsamples; ++k) {
                int predicted = act[(npy_intp)i * nsamples + k] - rep[(npy_intp)j * nsamples + k];
                if (predicted > 1) predicted = 1;
                else if (predicted < -1) predicted = -1;
                error += abs(gene[k] - predicted);
            }
            result[count] = (error * 1.0) / (nsamples * 1.0);
            ia[count] = i + 1;
            ir[count] = j + 1;
            ++count;
        }
    }
    return Py_BuildValue("OOOn", result_obj, ia_obj, ir_obj, count);
}

static PyMethodDef ComblicornMethods[] = {
    {"combLicorn", py_combLicorn, METH_VARARGS, "Calculate combLicorn"},
    {"combLicornDisjoint", py_combLicornDisjoint, METH_VARARGS, "Score disjoint regulator pairs; return buffers and valid count"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef comblicornmodule = {
    PyModuleDef_HEAD_INIT,
    "comblicorn",
    NULL,
    -1,
    ComblicornMethods
};

PyMODINIT_FUNC PyInit_comblicorn(void) {
    import_array();
    return PyModule_Create(&comblicornmodule);
}





