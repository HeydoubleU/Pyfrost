#ifndef PYFROST_H
#define PYFROST_H


#include "PyfrostExport.h"
#include <Amino/Core/Ptr.h>
#include <Bifrost/Object/Object.h>
#include <Bifrost/Math/Types.h>

#include <Amino/Core/Any.h>
#include <Amino/Core/Array.h>
#include <Amino/Core/String.h>
#include <Amino/Cpp/Annotate.h>

namespace Pyfrost {
    namespace Internal {
        PYFROST_DECL
            void pyfrost(
                Bifrost::Object& input AMINO_ANNOTATE("Amino::InOut outName=output"),
                const Amino::Array<Amino::String>& properties,
                const long long execution,
                const Amino::String& script,
                bool& success
            )
            AMINO_ANNOTATE("Amino::Node");

        PYFROST_DECL
            void pyfrost(
                Bifrost::Object& input AMINO_ANNOTATE("Amino::InOut outName=output"),
                const bool properties,
                const long long execution,
                const Amino::String& script,
                bool& success
            )
            AMINO_ANNOTATE("Amino::Node Amino::DefaultOverload");

        PYFROST_DECL
            void finalize_pyfrost(
                const bool finalize,
                bool& pass
            )
            AMINO_ANNOTATE("Amino::Node");
    } // namespace Internal
} // namespace Pyfrost

#endif // PYFROST_H


// TODO --------------------------
//      2D/3D array IO
//      python's array module or numpy for fast arrays
//      option to specify output data type ie: python int -> uint
//      option for overriding python interpreter path
//      file path as script input