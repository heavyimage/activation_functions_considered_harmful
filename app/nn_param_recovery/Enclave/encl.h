#ifndef _ENCLAVE_H_
#define _ENCLAVE_H_

#include <math.h>
#include <ctime>
#include <stdlib.h>
#include "string.h"
#include "stdlib.h"
#include "stdio.h"
#include "sgx_trts.h"
#include "encl_t.h"

#if defined(__cplusplus)
extern "C" {
#endif

int printf(const char* fmt, ...);
float perform_inference(int num_runs);

#if defined(__cplusplus)
}
#endif

#endif /* !_ENCLAVE_H_ */

