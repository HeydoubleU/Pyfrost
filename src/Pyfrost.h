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
                Amino::Ptr<Bifrost::Object>& input AMINO_ANNOTATE("Amino::InOut outName=output"),
                const Amino::String& script,
                const bool global,
                const bool consume_input,
                bool& success
            )
            AMINO_ANNOTATE("Amino::Node");

        // PYFROST_DECL void finalize_pyfrost(const bool finalize, bool& pass) AMINO_ANNOTATE("Amino::Node");
    } // namespace Internal
} // namespace Pyfrost

#endif // PYFROST_H